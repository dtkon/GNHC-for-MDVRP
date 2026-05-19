from typing import Callable, Optional, Union, overload
import math
import abc
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Policy(abc.ABC, nn.Module):
    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        in_query_dim: Optional[int],
        in_key_dim: Optional[int],
        in_val_dim: Optional[int],
        out_dim: int,
        only_score: bool = False,
    ) -> None:
        '''
        in_query_dim: None means q won't be linear projected.

        only_score: if only compute attention score.
        '''
        super().__init__()

        hidden_dim = out_dim // n_heads

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.in_query_dim = in_query_dim
        self.in_key_dim = in_key_dim
        self.in_val_dim = in_val_dim

        self.only_score = only_score

        # self.norm_factor = 1 / math.sqrt(hidden_dim)  # See Attention is all you need

        self.W_query = None
        self.W_key = None
        self.W_val = None
        self.W_out = None

        if in_query_dim is not None:
            self.W_query = nn.Parameter(torch.zeros(n_heads, in_query_dim, hidden_dim))
        if in_key_dim is not None:
            self.W_key = nn.Parameter(torch.zeros(n_heads, in_key_dim, hidden_dim))
        if in_val_dim is not None and not only_score:
            self.W_val = nn.Parameter(torch.zeros(n_heads, in_val_dim, hidden_dim))
        if not only_score:
            self.W_out = nn.Parameter(torch.zeros(n_heads, hidden_dim, out_dim))

    @staticmethod
    def compute(
        n_heads: int,
        hidden_dim: int,
        out_dim: int,
        q: Tensor,
        k: Tensor,
        v: Optional[Tensor] = None,
        W_query: Optional[Tensor] = None,
        W_key: Optional[Tensor] = None,
        W_val: Optional[Tensor] = None,
        W_out: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        with_norm: bool = True,
        only_score: bool = False,
    ) -> Tensor:
        batch_size, n_query, in_que_dim = q.size()
        _, n_key, in_key_dim = k.size()

        if v is not None:
            _, n_val, in_val_dim = v.size()
            assert n_key == n_val

        if W_query is not None:
            qflat = q.contiguous().view(
                -1, in_que_dim
            )  # (batch_size * n_query, in_que_dim)
            shp_Q = (n_heads, batch_size, n_query, hidden_dim)

            # Calculate queries, (n_heads, batch_size, n_query, hidden_dim)
            Q = torch.matmul(qflat, W_query).view(shp_Q)
            # self.W_query: (n_heads, in_que_dim, hidden_dim)
            # Q_before_view: (n_heads, batch_size * n_query, hidden_dim)
        else:
            assert in_que_dim == out_dim
            Q = q.view(batch_size, n_query, hidden_dim, n_heads).permute(3, 0, 1, 2)

        # Calculate keys and values (n_heads, batch_size, n_key, hidden_dim)
        shp_KV = (n_heads, batch_size, n_key, hidden_dim)

        if W_key is not None:
            kflat = k.contiguous().view(
                -1, in_key_dim
            )  # (batch_size * n_key, in_key_dim)
            K = torch.matmul(kflat, W_key).view(shp_KV)
        else:
            assert in_key_dim == out_dim
            K = k.view(batch_size, n_key, hidden_dim, n_heads).permute(3, 0, 1, 2)

        if v is not None:
            if W_val is not None:
                vflat = v.contiguous().view(-1, in_val_dim)
                V = torch.matmul(vflat, W_val).view(shp_KV)
            else:
                assert in_val_dim == out_dim
                V = v.view(batch_size, n_val, hidden_dim, n_heads).permute(3, 0, 1, 2)

        # Calculate compatibility (n_heads, batch_size, n_query, n_key)
        compatibility = torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None, :, None, :]  # (batch_size, n_key)
                mask = mask.expand(
                    n_heads, -1, n_query, -1
                )  # (n_heads, batch_size, n_query, n_key)
            elif mask.dim() == 3:
                mask = mask[None, :, :, :]  # (batch_size, n_query, n_key)
                mask = mask.expand(
                    n_heads, -1, -1, -1
                )  # (n_heads, batch_size, n_query, n_key)
            else:
                raise NotImplementedError
            compatibility[mask] = -float('inf')

        if only_score and not with_norm:
            return compatibility

        norm_factor = 1 / math.sqrt(hidden_dim)
        compatibility = norm_factor * compatibility

        if only_score and with_norm:
            return compatibility

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)  # (n_heads, batch_size, n_query, hidden_dim)

        assert W_out is not None

        out = torch.mm(
            heads.permute(1, 2, 0, 3)  # (batch_size, n_query, n_heads, hidden_dim)
            .contiguous()
            .view(
                -1, n_heads * hidden_dim
            ),  # (batch_size * n_query, n_heads * hidden_dim)
            W_out.view(-1, out_dim),  # (n_heads * hidden_dim, out_dim)
        ).view(batch_size, n_query, out_dim)

        return out

    __call__: Callable[..., Tensor]

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        with_norm: bool = True,
    ) -> Tensor:
        '''
        q: (batch_size, n_query, in_que_dim)

        k: (batch_size, n_key, in_key_dim)

        v: (batch_size, n_key, in_val_dim)

        mask: (batch_size, n_key)
        '''
        if self.only_score:  # calculate attention score
            assert v is None

        return self.compute(
            self.n_heads,
            self.hidden_dim,
            self.out_dim,
            q,
            k,
            v,
            self.W_query,
            self.W_key,
            self.W_val,
            self.W_out,
            mask,
            with_norm,
            self.only_score,
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()
        self.MHA = MultiHeadAttention(
            n_heads, input_dim, input_dim, input_dim, input_dim
        )

    __call__: Callable[..., Tensor]

    def forward(self, q: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.MHA(q, q, q, mask=mask)


class MultiHeadSelfAttentionScore(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()
        self.MHA = MultiHeadAttention(n_heads, input_dim, input_dim, None, input_dim)

    __call__: Callable[..., Tensor]

    def forward(self, q: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.MHA(q, q, mask=mask)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        feed_forward_dim: int = 64,
        embedding_dim: int = 64,
        output_dim: int = 1,
        p_dropout: float = 0.01,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        self.ReLU = nn.ReLU(inplace=True)

    __call__: Callable[..., Tensor]

    def forward(self, input: Tensor) -> Tensor:
        result = self.ReLU(self.fc1(input))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result)
        return result


class Normalization(nn.Module):
    def __init__(self, input_dim: int, normalization: str) -> None:
        super().__init__()

        self.normalization = normalization

        if normalization == 'none':
            pass
        elif normalization != 'layer':
            normalizer_class = {'batch': nn.BatchNorm1d, 'instance': nn.InstanceNorm1d}[
                normalization
            ]
            self.normalizer = normalizer_class(input_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    # def init_parameters(self) -> None:
    #    for param in self.parameters():
    #        stdv = 1.0 / math.sqrt(param.size(-1))
    #        param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., Tensor]

    def forward(self, input: Tensor) -> Tensor:
        if self.normalization == 'none':
            return input
        elif self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                input.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif self.normalization == 'batch':
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif self.normalization == 'instance':
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise NotImplementedError('Unknown normalizer type')


class FFNormSubLayer(nn.Module):
    def __init__(
        self, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.FF = (
            nn.Sequential(
                nn.Linear(input_dim, feed_forward_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, input_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(input_dim, input_dim)
        )

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., Tensor]

    def forward(self, input: Tensor) -> Tensor:
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class EncodingBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        feed_forward_hidden: int,
        normalization: str,
    ) -> None:
        super().__init__()
        self.MHA = MultiHeadSelfAttention(n_heads, input_dim)
        self.norm = Normalization(input_dim, normalization)
        self.FFnorm = FFNormSubLayer(input_dim, feed_forward_hidden, normalization)

    __call__: Callable[..., Union[Tensor, tuple[Tensor, Tensor]]]

    @overload
    def forward(self, input: Tensor) -> Tensor: ...

    @overload
    def forward(self, input: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]: ...

    def forward(self, input: Union[Tensor, tuple[Tensor, Tensor]]):
        if isinstance(input, tuple):
            mask = input[1]
            input = input[0]
            return self.FFnorm(self.norm(input + self.MHA(input, mask=mask))), mask
        else:
            return self.FFnorm(self.norm(input + self.MHA(input)))


class CrossAttention_EncodingBlock(nn.Module):
    def __init__(
        self, n_heads: int, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.MHA_1 = MultiHeadAttention(
            n_heads, input_dim, input_dim, input_dim, input_dim
        )
        self.norm_1 = Normalization(input_dim, normalization)
        self.FFskip_1 = FFNormSubLayer(input_dim, feed_forward_hidden, normalization)

        self.MHA_2 = MultiHeadAttention(
            n_heads, input_dim, input_dim, input_dim, input_dim
        )
        self.norm_2 = Normalization(input_dim, normalization)
        self.FFskip_2 = FFNormSubLayer(input_dim, feed_forward_hidden, normalization)

    __call__: Callable[..., tuple[Tensor, ...]]

    def forward(self, input: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        Z = input[0]
        H = input[1]
        if len(input) == 4:
            mask_for_Z = input[2]
            mask_for_H = input[3]
        else:
            mask_for_Z = None
            mask_for_H = None

        next_Z = self.FFskip_1(self.norm_1(Z + self.MHA_1(Z, H, H, mask=mask_for_H)))
        next_H = self.FFskip_2(
            self.norm_2(H + self.MHA_2(H, next_Z, next_Z, mask=mask_for_Z))
        )

        if len(input) == 4:
            return next_Z, next_H, mask_for_Z, mask_for_H  # type: ignore
        else:
            return next_Z, next_H


class AM_Encoder_CVRP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_hidden: int,
        n_heads: int,
        n_blocks: int,
        normalization: str,
    ) -> None:
        super().__init__()

        self.customer_embedder = nn.Linear(3, embedding_dim)
        self.depot_embedder = nn.Linear(2, embedding_dim)

        if n_blocks > 0:
            self.encoding_blocks: Optional[nn.Module] = nn.Sequential(
                *(
                    EncodingBlock(
                        n_heads, embedding_dim, feed_forward_hidden, normalization
                    )
                    for _ in range(n_blocks)
                )
            )
        else:
            self.encoding_blocks = None

    __call__: Callable[..., Tensor]

    def forward(self, input: Tensor, depot_num: int = 1) -> Tensor:
        '''
        input: graph[batch_size, problem_size, 3(x,y,d)], depot(first node)'s demand=0

        return: (batch_size, problem_size, embedding_dim)
        '''
        customers = input[:, depot_num:, :]
        depot = input[:, :depot_num, :2]

        cus_emb = self.customer_embedder(customers)
        dep_emb = self.depot_embedder(depot)

        init_embedding = torch.cat((dep_emb, cus_emb), 1)

        if self.encoding_blocks is not None:
            embedding = self.encoding_blocks(init_embedding)
        else:
            embedding = init_embedding

        return embedding


class AM_Encoder_TSP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_hidden: int,
        n_heads: int,
        n_blocks: int,
        normalization: str,
    ) -> None:
        super().__init__()

        self.customer_embedder = nn.Linear(2, embedding_dim)

        if n_blocks > 0:
            self.encoding_blocks: Optional[nn.Module] = nn.Sequential(
                *(
                    EncodingBlock(
                        n_heads, embedding_dim, feed_forward_hidden, normalization
                    )
                    for _ in range(n_blocks)
                )
            )
        else:
            self.encoding_blocks = None

    __call__: Callable[..., Tensor]

    def forward(self, input: Tensor, depot_num: Optional[int] = 0) -> Tensor:
        '''
        input: graph[batch_size, problem_size, 2(x,y)]

        return: (batch_size, problem_size, embedding_dim)
        '''
        init_embedding = self.customer_embedder(input)

        if self.encoding_blocks is not None:
            embedding = self.encoding_blocks(init_embedding)
        else:
            embedding = init_embedding

        return embedding


class AM_Decoder_Precompute(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.enc_key_proj = nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.enc_val_proj = nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.enc_key_for_glimpse_proj = nn.Parameter(
            torch.zeros(embedding_dim, embedding_dim)
        )

    @staticmethod
    def compute(
        graph_embedding: Tensor,
        enc_key_proj: Tensor,
        enc_val_proj: Tensor,
        enc_key_for_glimpse_proj: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        proj_key = F.linear(graph_embedding, enc_key_proj)
        proj_val = F.linear(graph_embedding, enc_val_proj)
        proj_key_for_glimpse = F.linear(graph_embedding, enc_key_for_glimpse_proj)

        return proj_key, proj_val, proj_key_for_glimpse

    __call__: Callable[..., tuple[Tensor, Tensor, Tensor]]

    def forward(self, graph_embedding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.compute(
            graph_embedding,
            self.enc_key_proj,
            self.enc_val_proj,
            self.enc_key_for_glimpse_proj,
        )


class AM_Decoder_construct(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int, context_dim: int) -> None:
        super().__init__()

        self.first_MHA = MultiHeadAttention(
            n_heads, context_dim, None, None, embedding_dim
        )

        self.second_SHA_score = MultiHeadAttention(
            1, None, None, None, embedding_dim, only_score=True
        )

    @staticmethod
    def compute(
        context: Tensor,
        proj_key: Tensor,
        proj_val: Tensor,
        proj_key_for_glimpse: Tensor,
        first_MHA: Callable[..., Tensor],
        second_SHA_score: Callable[..., Tensor],
        mask: Optional[Tensor] = None,
        C: float = 10,
        select_type: str = 'sample',
        fixed_next_node: Optional[Tensor] = None,
        temperature: float = 1.0,
        cross_prob: bool = False,
    ) -> tuple[Tensor, Tensor]:
        batch_size, ref_num, _ = proj_key.size()
        query_num = context.size(1)

        glimpse = first_MHA(context, proj_key, proj_val, mask=mask)

        compatibility = (
            torch.tanh(second_SHA_score(glimpse, proj_key_for_glimpse)) * C
        ).squeeze(0) / temperature
        # (batch_size, query_num, ref_num)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand(-1, query_num, -1)
            compatibility[mask] = -float('inf')

        if not cross_prob:
            prob = F.softmax(compatibility, dim=-1)
            log_p = F.log_softmax(
                compatibility, dim=-1
            )  # (batch_size, query_num, ref_num)

            if fixed_next_node is None:
                if select_type == 'greedy':
                    next_node = (
                        prob.view(-1, ref_num).max(1)[1].reshape(batch_size, query_num)
                    )  # (batch_size, query_num)
                elif select_type == 'sample':
                    next_node = (
                        prob.view(-1, ref_num)
                        .multinomial(1)
                        .reshape(batch_size, query_num)
                    )  # (batch_size, query_num)
                elif select_type == 'distrib':
                    return prob, compatibility
                else:
                    raise NotImplementedError
            else:
                next_node = fixed_next_node

            arange = torch.arange(batch_size * query_num)
            sel_log_p = log_p.view(-1, ref_num)[arange, next_node.view(-1)].reshape(
                batch_size, query_num
            )  # (batch_size, query_num)
        else:
            prob = F.softmax(compatibility.view(batch_size, -1), dim=-1)
            log_p = F.log_softmax(
                compatibility.view(batch_size, -1), dim=-1
            )  # (batch_size, query_num*ref_num)

            if fixed_next_node is None:
                if select_type == 'greedy':
                    next_node_sel = prob.max(1)[1]  # (batch_size,)
                elif select_type == 'sample':
                    next_node_sel = prob.multinomial(1).view(-1)  # (batch_size,)
                elif select_type == 'distrib':
                    return prob, compatibility
                else:
                    raise NotImplementedError

                vehicle_sel = torch.div(
                    next_node_sel, ref_num, rounding_mode='trunc'
                )  # (batch_size,)
                customer_sel = next_node_sel % ref_num

                next_node = torch.stack((vehicle_sel, customer_sel), dim=1)
            else:
                next_node = fixed_next_node  # (batch_size, 2)

            arange = torch.arange(batch_size)
            sel_log_p = log_p[arange, next_node_sel]  # (batch_size,)

        return next_node, sel_log_p

    __call__: Callable[..., tuple[Tensor, Tensor]]

    def forward(
        self,
        context: Tensor,
        proj_key: Tensor,
        proj_val: Tensor,
        proj_key_for_glimpse: Tensor,
        mask: Optional[Tensor] = None,
        C: float = 10,
        select_type: str = 'sample',
        fixed_next_node: Optional[Tensor] = None,
        temperature: float = 1.0,
        cross_prob: bool = False,
    ) -> tuple[Tensor, Tensor]:
        '''
        proj_key: projected key, (batch_size, key_num/val_num, embedding_dim)

        context: query, (batch_size, query_num, context_dim)

        mask: bool(batch_size, problem_size) or (batch_size, query_num, problem_size)

        return: [next_nodes(batch_size, query_num or 2), log_prob(batch_size, query_num or None)]
                or [prob_distrib(batch_size, query_num(, or *)ref_num),
                    compatibility(batch_size, query_num, ref_num)]
        '''
        return self.compute(
            context,
            proj_key,
            proj_val,
            proj_key_for_glimpse,
            self.first_MHA,
            self.second_SHA_score,
            mask,
            C,
            select_type,
            fixed_next_node,
            temperature,
            cross_prob,
        )


class AM_Decoder(nn.Module):
    def __init__(self, n_heads: int, embedding_dim: int, context_dim: int) -> None:
        super().__init__()

        self.first_MHA = MultiHeadAttention(
            n_heads, context_dim, None, None, embedding_dim
        )

        self.second_SHA_score = MultiHeadAttention(
            1, None, None, None, embedding_dim, only_score=True
        )

    @staticmethod
    def compute(
        context: Tensor,
        proj_key: Tensor,
        proj_val: Tensor,
        proj_key_for_glimpse: Tensor,
        first_MHA: Callable[..., Tensor],
        second_SHA_score: Callable[..., Tensor],
        mask: Optional[Tensor] = None,
        mask_for_glimpse: Optional[Tensor] = None,
        C: float = 10.0,
        temperature: float = 1.0,
        cross_prob: bool = False,
    ) -> Tensor:
        batch_size, query_num, _ = context.size()

        if mask_for_glimpse is None:
            mask_for_glimpse = mask

        glimpse = first_MHA(context, proj_key, proj_val, mask=mask_for_glimpse)

        compatibility = (
            torch.tanh(second_SHA_score(glimpse, proj_key_for_glimpse)) * C
        ).squeeze(0) / temperature
        # (batch_size, query_num, ref_num)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand(-1, query_num, -1)
            compatibility[mask] = -float('inf')

        if not cross_prob:
            prob = F.softmax(compatibility, dim=-1)  # (batch_size, query_num, ref_num)
            # log_p = F.log_softmax(compatibility, dim=-1)  # (batch_size, query_num, ref_num)
        else:
            prob = F.softmax(compatibility.view(batch_size, -1), dim=-1).view(
                batch_size, query_num, -1
            )

        return prob

    __call__: Callable[..., Tensor]

    def forward(
        self,
        context: Tensor,
        proj_key: Tensor,
        proj_val: Tensor,
        proj_key_for_glimpse: Tensor,
        mask: Optional[Tensor] = None,
        mask_for_glimpse: Optional[Tensor] = None,
        C: float = 10,
        temperature: float = 1.0,
        cross_prob: bool = False,
    ) -> Tensor:
        '''
        proj_key: projected key, (batch_size, key_num/val_num, embedding_dim)

        context: query, (batch_size, query_num, context_dim)

        mask: bool(batch_size, problem_size) or (batch_size, query_num, problem_size)

        return: prob(batch_size, query_num, ref_num)
        '''
        return self.compute(
            context,
            proj_key,
            proj_val,
            proj_key_for_glimpse,
            self.first_MHA,
            self.second_SHA_score,
            mask,
            mask_for_glimpse,
            C,
            temperature,
            cross_prob,
        )


# https://github.com/kaist-silab/equity-transformer/blob/main/nets/positional_encoding.py
class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    __call__: Callable[..., torch.Tensor]

    def forward(
        self, batch_size: int, seq_len: int, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if mask is None:
            return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
            # [batch_size = 128, seq_len = 30, d_model = 512]
            # it will add with tok_emb : [128, 30, 512]

        assert mask.shape == (batch_size, seq_len)

        cum_counts = (~mask).long().cumsum(dim=1)
        indices = cum_counts - 1

        enc = self.encoding[indices]  # [batch_size, seq_len, d_model]
        return enc.masked_fill(mask.unsqueeze(-1), 0.0)
