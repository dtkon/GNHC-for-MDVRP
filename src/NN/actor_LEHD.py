from typing import Optional, Union
import torch
from torch import Tensor, nn

from . import layers


class LEHD_Encoder(layers.Policy):
    def __init__(
        self,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        n_heads: int = 8,
        problem: str = 'vrp',
    ) -> None:
        super().__init__()

        if problem == 'vrp':
            self.graph_encoder: Union[layers.AM_Encoder_CVRP, layers.AM_Encoder_TSP] = (
                layers.AM_Encoder_CVRP(
                    embedding_dim, feed_forward_hidden, n_heads, 1, 'none'
                )
            )
        elif problem == 'tsp':
            self.graph_encoder = layers.AM_Encoder_TSP(
                embedding_dim, feed_forward_hidden, n_heads, 1, 'none'
            )
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(self, problems: Tensor, depot_num: int = 1) -> tuple[Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d) or 2(x,y)]
        '''

        # encoder
        graph_embed = self.graph_encoder(
            problems, depot_num=depot_num
        )  # (batch_size, problem_size, embedding_dim)

        return (graph_embed,)


class LEHD_Decoder(layers.Policy):
    def __init__(
        self,
        n_heads: int = 8,
        n_blocks: int = 6,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        normalization: str = 'layer',
        problem: str = 'vrp',
    ) -> None:
        super().__init__()

        if problem == 'tsp':
            self.start_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.end_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif problem == 'vrp':
            self.start_proj = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
            self.end_proj = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
        else:
            raise NotImplementedError

        self.encoding_blocks: nn.Module = nn.Sequential(
            *(
                layers.EncodingBlock(
                    n_heads, embedding_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_blocks)
            )
        )

        if problem == 'tsp':
            self.out_proj = nn.Linear(embedding_dim, 1, bias=False)
        elif problem == 'vrp':
            self.out_proj = nn.Linear(embedding_dim, 2, bias=False)
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(
        self,
        graph_embed: Tensor,
        start_index: Tensor,
        end_index: Tensor,
        available_index: Tensor,
        unavailable_mask: Optional[Tensor] = None,
        remain_capacity: Optional[Tensor] = None,
        infeasible_mask: Optional[Tensor] = None,
    ) -> Tensor:
        '''
        graph_embed: (batch_size, problem_size, embedding_dim)

        start_index, end_index: (batch_size, 1)

        available_index: (batch_size, node_num)

        unavailable_mask: (batch_size, node_num), mask for available embed

        remain_capacity: (batch_size, 1)

        infeasible_mask: No need for tsp. For vrp, shape is (batch_size, node_num, 2)
        '''
        batch_size, problem_size, embed_dim = graph_embed.size()

        batch_arange = torch.arange(batch_size, device=graph_embed.device)

        start_embed = graph_embed[batch_arange, start_index.view(-1)]
        end_embed = graph_embed[batch_arange, end_index.view(-1)]
        available_embed = graph_embed.gather(
            1, available_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )

        if remain_capacity is not None:
            start_embed = torch.cat((start_embed, remain_capacity), dim=1)
            end_embed = torch.cat((end_embed, remain_capacity), dim=1)

        start_embed_proj: Tensor = self.start_proj(start_embed)
        end_embed_proj: Tensor = self.end_proj(end_embed)

        all_embed = torch.cat(
            (
                start_embed_proj.unsqueeze(1),
                end_embed_proj.unsqueeze(1),
                available_embed,
            ),
            dim=1,
        )

        endpoint_mask = torch.zeros(
            (batch_size, 2), dtype=torch.bool, device=graph_embed.device
        )

        if unavailable_mask is None:
            unavailable_mask = torch.zeros_like(available_index, dtype=torch.bool)

        cated_unavailable_mask = torch.cat((endpoint_mask, unavailable_mask), dim=1)

        encoding_result: Tensor
        encoding_result, _ = self.encoding_blocks((all_embed, cated_unavailable_mask))

        result_proj: Tensor = self.out_proj(
            encoding_result[:, 2:, :]
        )  # (batch_size, node_num, 1 or 2)

        if infeasible_mask is not None:
            # unavailable_mask = unavailable_mask.unsqueeze(2).expand(-1, -1, 2)
            result_proj[infeasible_mask] = -float('inf')  # (batch_size, node_num, 2)
            prob = (
                result_proj.view(batch_size, -1).softmax(1).view(batch_size, -1, 2)
            )  # (batch_size, node_num, 2)
        else:
            result_proj = result_proj.squeeze(2)
            result_proj[unavailable_mask] = -float('inf')  # (batch_size, node_num)
            prob = result_proj.softmax(1)  # (batch_size, node_num)

        return prob


class LEHD_CrossAtt_Encoder(layers.Policy):
    def __init__(self, embedding_dim: int = 128, problem: str = 'vrp') -> None:
        super().__init__()

        if problem == 'vrp':
            self.graph_encoder: Union[layers.AM_Encoder_CVRP, layers.AM_Encoder_TSP] = (
                layers.AM_Encoder_CVRP(embedding_dim, 1, 1, 0, 'none')
            )
        elif problem == 'tsp':
            self.graph_encoder = layers.AM_Encoder_TSP(embedding_dim, 1, 1, 0, 'none')
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(self, problems: Tensor, depot_num: int = 1) -> tuple[Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d) or 2(x,y)]
        '''

        # encoder
        graph_embed = self.graph_encoder(
            problems, depot_num=depot_num
        )  # (batch_size, problem_size, embedding_dim)

        return (graph_embed,)


class LEHD_CrossAtt_Decoder(layers.Policy):
    def __init__(
        self,
        n_heads: int = 8,
        n_blocks: int = 6,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        normalization: str = 'layer',
        problem: str = 'vrp',
    ) -> None:
        super().__init__()

        if problem == 'tsp':
            self.start_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.end_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif problem == 'vrp':
            self.start_proj = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
            self.end_proj = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
        else:
            raise NotImplementedError

        self.encoding_blocks: nn.Module = nn.Sequential(
            *(
                layers.CrossAttention_EncodingBlock(
                    n_heads, embedding_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_blocks)
            )
        )

        if problem == 'tsp':
            self.out_proj = nn.Linear(embedding_dim, 1, bias=False)
        elif problem == 'vrp':
            self.out_proj = nn.Linear(embedding_dim, 2, bias=False)
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(
        self,
        graph_embed: Tensor,
        start_index: Tensor,
        end_index: Tensor,
        available_index: Tensor,
        unavailable_mask: Optional[Tensor] = None,
        remain_capacity: Optional[Tensor] = None,
        infeasible_mask: Optional[Tensor] = None,
    ) -> Tensor:
        '''
        graph_embed: (batch_size, problem_size, embedding_dim)

        start_index, end_index: (batch_size, 1)

        available_index: (batch_size, node_num)

        unavailable_mask: (batch_size, node_num), mask for available embed

        remain_capacity: (batch_size, 1)

        infeasible_mask: No need for tsp. For vrp, shape is (batch_size, node_num, 2)
        '''
        batch_size, problem_size, embed_dim = graph_embed.size()

        batch_arange = torch.arange(batch_size, device=graph_embed.device)

        start_embed = graph_embed[batch_arange, start_index.view(-1)]
        end_embed = graph_embed[batch_arange, end_index.view(-1)]
        available_embed = graph_embed.gather(
            1, available_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )

        if remain_capacity is not None:
            start_embed = torch.cat((start_embed, remain_capacity), dim=1)
            end_embed = torch.cat((end_embed, remain_capacity), dim=1)

        start_embed_proj: Tensor = self.start_proj(start_embed)
        end_embed_proj: Tensor = self.end_proj(end_embed)

        init_Z = torch.cat(
            (start_embed_proj.unsqueeze(1), end_embed_proj.unsqueeze(1)),
            dim=1,
        )

        init_H = torch.cat(
            (
                start_embed_proj.unsqueeze(1),
                end_embed_proj.unsqueeze(1),
                available_embed,
            ),
            dim=1,
        )

        endpoint_mask = torch.zeros(
            (batch_size, 2), dtype=torch.bool, device=graph_embed.device
        )

        if unavailable_mask is None:
            unavailable_mask = torch.zeros_like(available_index, dtype=torch.bool)

        cated_unavailable_mask = torch.cat((endpoint_mask, unavailable_mask), dim=1)

        final_H: Tensor
        _, final_H, *_ = self.encoding_blocks(
            (init_Z, init_H, None, cated_unavailable_mask)
        )

        result_proj: Tensor = self.out_proj(
            final_H[:, 2:, :]
        )  # (batch_size, node_num, 1 or 2)

        if infeasible_mask is not None:
            # unavailable_mask = unavailable_mask.unsqueeze(2).expand(-1, -1, 2)
            result_proj[infeasible_mask] = -float('inf')  # (batch_size, node_num, 2)
            prob = (
                result_proj.view(batch_size, -1).softmax(1).view(batch_size, -1, 2)
            )  # (batch_size, node_num, 2)
        else:
            result_proj = result_proj.squeeze(2)
            result_proj[unavailable_mask] = -float('inf')  # (batch_size, node_num)
            prob = result_proj.softmax(1)  # (batch_size, node_num)

        return prob


class LEHD_CrossAtt_Decoder_MDVRP_lpOut(layers.Policy):
    def __init__(
        self,
        n_heads: int = 8,
        n_blocks: int = 6,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        normalization: str = 'layer',
        max_depot_num: int = 100,
    ) -> None:
        super().__init__()

        # start context: start embed, remaining capacity
        self.start_context_proj = nn.Linear(embedding_dim + 1, embedding_dim)
        # end context: depot embed, remaining capacity
        self.end_context_proj = nn.Linear(embedding_dim + 1, embedding_dim)

        self.pos_enc = layers.PostionalEncoding(embedding_dim, max_depot_num)

        self.encoding_blocks: nn.Module = nn.Sequential(
            *(
                layers.CrossAttention_EncodingBlock(
                    n_heads, embedding_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_blocks)
            )
        )

        self.out_proj = nn.Linear(embedding_dim * 3, 2, bias=False)

        self.init_parameters()

    def forward(
        self,
        graph_embed: Tensor,
        start_index: Tensor,
        depot_index: Tensor,
        available_index: Tensor,
        unavailable_mask: Optional[Tensor],
        remain_capacity: Tensor,
        capacity: Tensor,
        lock_depot: Optional[Tensor],
        infeasible_mask: Tensor,
    ) -> Tensor:
        '''
        graph_embed: (batch_size, problem_size, embedding_dim)

        start_index, end_index, depot_index: (batch_size, depot_num)

        available_index: (batch_size, node_num)

        unavailable_mask: (batch_size, node_num), mask for available embed

        remain_capacity: (batch_size, depot_num)

        lock_depot: (batch_size, depot_num)

        infeasible_mask: (batch_size, depot_num, node_num, 2)
        '''
        batch_size, _, embed_dim = graph_embed.size()
        depot_num = start_index.size(1)
        node_num = available_index.size(1)

        start_embed = graph_embed.gather(
            1, start_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )  # (batch_size, depot_num, embed_dim)
        depot_embed = graph_embed.gather(
            1, depot_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )
        available_embed = graph_embed.gather(
            1, available_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )  # (batch_size, node_num, embed_dim)

        start_context = torch.cat(
            (start_embed, (remain_capacity / capacity).unsqueeze(2)),
            dim=2,
        )

        end_context = torch.cat(
            (depot_embed, (remain_capacity / capacity).unsqueeze(2)),
            dim=2,
        )

        pos_embed = self.pos_enc(
            batch_size, depot_num, lock_depot
        )  # (batch_size, depot_num, embed_dim)

        start_context_embed: Tensor = pos_embed + self.start_context_proj(
            start_context
        )  # (batch_size, depot_num, embed_dim)

        end_context_embed: Tensor = pos_embed + self.end_context_proj(
            end_context
        )  # (batch_size, depot_num, embed_dim)

        init_Z = torch.cat((start_context_embed, end_context_embed), dim=1)

        if lock_depot is None:
            mask_for_Z = None

            endpoint_mask = torch.zeros(
                (batch_size, depot_num * 2), dtype=torch.bool, device=graph_embed.device
            )
        else:
            # when one instance in a batch is unable to re-constructed, there will be all true in its lock_depot.
            mask_for_Z = lock_depot.repeat(1, 2)  # (batch_size, depot_num*2)
            mask_for_Z[:, 0][mask_for_Z.all(1)] = False

            endpoint_mask = mask_for_Z

        init_H = torch.cat(
            (start_context_embed, end_context_embed, available_embed), dim=1
        )

        if unavailable_mask is None:
            unavailable_mask = torch.zeros_like(available_index, dtype=torch.bool)

        cated_unavailable_mask_for_H = torch.cat(
            (endpoint_mask, unavailable_mask), dim=1
        )

        final_Z: Tensor  # (batch_size, depot_num*2, embed_dim)
        final_H: Tensor  # (batch_size, depot_num*2+node_num, embed_dim)
        final_Z, final_H, *_ = self.encoding_blocks(
            (init_Z, init_H, mask_for_Z, cated_unavailable_mask_for_H)
        )

        final_Z_start = final_Z[:, :depot_num, :]
        final_Z_end = final_Z[:, depot_num:, :]
        combine_final_Z = torch.cat(
            (final_Z_start, final_Z_end), dim=2
        )  # (batch_size, depot_num, embed_dim*2)

        to_combine_Z = combine_final_Z.unsqueeze(2).expand(-1, -1, node_num, -1)
        to_combine_H = (
            final_H[:, depot_num * 2 :, :].unsqueeze(1).expand(-1, depot_num, -1, -1)
        )
        compatability: Tensor = self.out_proj(
            torch.cat((to_combine_Z, to_combine_H), dim=3)
        )  # (batch_size, depot_num, node_num, 2)

        # unavailable_mask = (
        #    unavailable_mask.unsqueeze(1).unsqueeze(3).expand(-1, depot_num, -1, 2)
        # )
        compatability[infeasible_mask] = -float('inf')
        prob = (
            compatability.view(batch_size, -1)
            .softmax(1)
            .view(batch_size, depot_num, node_num, 2)
        )

        return prob


class LEHD_CrossAtt_Decoder_MDVRP_attOut(layers.Policy):
    def __init__(
        self,
        n_heads: int = 8,
        n_blocks: int = 6,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        normalization: str = 'layer',
        max_depot_num: int = 100,
    ) -> None:
        super().__init__()

        # start context: start embed, remaining capacity
        self.start_context_proj = nn.Linear(embedding_dim + 1, embedding_dim)
        # end context: depot embed, remaining capacity
        self.end_context_proj = nn.Linear(embedding_dim + 1, embedding_dim)

        self.pos_enc = layers.PostionalEncoding(embedding_dim, max_depot_num)

        self.encoding_blocks: nn.Module = nn.Sequential(
            *(
                layers.CrossAttention_EncodingBlock(
                    n_heads, embedding_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_blocks)
            )
        )

        self.SHA_score = layers.MultiHeadAttention(
            1, None, None, None, embedding_dim, only_score=True
        )

        self.init_parameters()

    def forward(
        self,
        graph_embed: Tensor,
        start_index: Tensor,
        depot_index: Tensor,
        available_index: Tensor,
        unavailable_mask: Optional[Tensor],
        remain_capacity: Tensor,
        capacity: Tensor,
        lock_depot: Optional[Tensor],
        infeasible_mask: Tensor,
        C: float = 10.0,
    ) -> Tensor:
        '''
        graph_embed: (batch_size, problem_size, embedding_dim)

        start_index, end_index, depot_index: (batch_size, depot_num)

        available_index: (batch_size, node_num)

        unavailable_mask: (batch_size, node_num), mask for available embed

        remain_capacity: (batch_size, depot_num)

        lock_depot: (batch_size, depot_num)

        infeasible_mask: (batch_size, depot_num, node_num, 2)
        '''
        batch_size, _, embed_dim = graph_embed.size()
        depot_num = start_index.size(1)
        node_num = available_index.size(1)

        start_embed = graph_embed.gather(
            1, start_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )  # (batch_size, depot_num, embed_dim)
        depot_embed = graph_embed.gather(
            1, depot_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )
        available_embed = graph_embed.gather(
            1, available_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )  # (batch_size, node_num, embed_dim)

        start_context = torch.cat(
            (start_embed, (remain_capacity / capacity).unsqueeze(2)),
            dim=2,
        )

        end_context = torch.cat(
            (depot_embed, (remain_capacity / capacity).unsqueeze(2)),
            dim=2,
        )

        pos_embed = self.pos_enc(
            batch_size, depot_num, lock_depot
        )  # (batch_size, depot_num, embed_dim)

        start_context_embed: Tensor = pos_embed + self.start_context_proj(
            start_context
        )  # (batch_size, depot_num, embed_dim)

        end_context_embed: Tensor = pos_embed + self.end_context_proj(
            end_context
        )  # (batch_size, depot_num, embed_dim)

        init_Z = torch.cat((start_context_embed, end_context_embed), dim=1)

        if lock_depot is None:
            mask_for_Z = None
            endpoint_mask = torch.zeros(
                (batch_size, depot_num * 2), dtype=torch.bool, device=graph_embed.device
            )
        else:
            # when one instance in a batch is unable to re-constructed, there will be all true in its lock_depot.
            mask_for_Z = lock_depot.repeat(1, 2)  # (batch_size, depot_num*2)
            mask_for_Z[:, 0][mask_for_Z.all(1)] = False

            endpoint_mask = mask_for_Z

        init_H = torch.cat(
            (start_context_embed, end_context_embed, available_embed), dim=1
        )

        if unavailable_mask is None:
            unavailable_mask = torch.zeros_like(available_index, dtype=torch.bool)

        cated_unavailable_mask_for_H = torch.cat(
            (endpoint_mask, unavailable_mask), dim=1
        )

        final_Z: Tensor  # (batch_size, depot_num*2, embed_dim)
        final_H: Tensor  # (batch_size, depot_num*2+node_num, embed_dim)
        final_Z, final_H, *_ = self.encoding_blocks(
            (init_Z, init_H, mask_for_Z, cated_unavailable_mask_for_H)
        )

        final_Z_start = final_Z[:, :depot_num, :]
        final_Z_end = final_Z[:, depot_num:, :]

        compatability_1 = (
            torch.tanh(self.SHA_score(final_Z_start, final_H[:, depot_num * 2 :, :]))
            * C
        ).squeeze(
            0
        )  # (batch_size, depot_num, node_num)
        compatability_2 = (
            torch.tanh(self.SHA_score(final_Z_end, final_H[:, depot_num * 2 :, :])) * C
        ).squeeze(
            0
        )  # (batch_size, depot_num, node_num)

        compatability = torch.stack((compatability_1, compatability_2), dim=3)

        compatability[infeasible_mask] = -float('inf')
        prob = (
            compatability.view(batch_size, -1)
            .softmax(1)
            .view(batch_size, depot_num, node_num, 2)
        )

        return prob


class LEHD_CrossAtt_Decoder_MDVRP_attOut_directAction(layers.Policy):
    def __init__(
        self,
        n_heads: int = 8,
        n_blocks: int = 6,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        normalization: str = 'layer',
        max_depot_num: int = 100,
    ) -> None:
        '''
        max_depot_num set to 1 for CVRP
        '''
        super().__init__()

        # start context: start embed, remaining capacity
        self.start_context_proj = nn.Linear(embedding_dim + 1, embedding_dim)
        # end context: depot embed, remaining capacity
        self.end_context_proj = nn.Linear(embedding_dim + 1, embedding_dim)

        if max_depot_num > 1:
            self.pos_enc: Optional[layers.PostionalEncoding] = layers.PostionalEncoding(
                embedding_dim, max_depot_num
            )
        else:
            self.pos_enc = None

        self.encoding_blocks: nn.Module = nn.Sequential(
            *(
                layers.CrossAttention_EncodingBlock(
                    n_heads, embedding_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_blocks)
            )
        )

        self.SHA_score = layers.MultiHeadAttention(
            1, embedding_dim * 2, None, None, embedding_dim, only_score=True
        )

        self.init_parameters()

    def forward(
        self,
        graph_embed: Tensor,
        start_index: Tensor,
        depot_index: Tensor,
        available_index: Tensor,
        unavailable_mask: Optional[Tensor],
        remain_capacity: Tensor,
        capacity: Tensor,
        lock_depot: Optional[Tensor],
        infeasible_mask: Tensor,
        C: float = 10.0,
    ) -> Tensor:
        '''
        graph_embed: (batch_size, problem_size, embedding_dim)

        start_index, end_index, depot_index: (batch_size, depot_num)

        available_index: (batch_size, depot_num+node_num)

        unavailable_mask: (batch_size, depot_num+node_num), mask for available embed

        remain_capacity: (batch_size, depot_num)

        lock_depot: (batch_size, depot_num)

        infeasible_mask: (batch_size, depot_num, depot_num+node_num)
        '''
        batch_size, _, embed_dim = graph_embed.size()
        depot_num = start_index.size(1)
        node_num = available_index.size(1) - depot_num

        start_embed = graph_embed.gather(
            1, start_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )  # (batch_size, depot_num, embed_dim)
        depot_embed = graph_embed.gather(
            1, depot_index.unsqueeze(2).expand(-1, -1, embed_dim)
        )
        available_embed = graph_embed.gather(
            1, available_index[:, depot_num:].unsqueeze(2).expand(-1, -1, embed_dim)
        )  # (batch_size, node_num, embed_dim)

        start_context = torch.cat(
            (start_embed, (remain_capacity / capacity).unsqueeze(2)),
            dim=2,
        )

        end_context = torch.cat(
            (depot_embed, (remain_capacity / capacity).unsqueeze(2)),
            dim=2,
        )

        start_context_embed: Tensor = self.start_context_proj(
            start_context
        )  # (batch_size, depot_num, embed_dim)

        end_context_embed: Tensor = self.end_context_proj(
            end_context
        )  # (batch_size, depot_num, embed_dim)

        if self.pos_enc is not None:
            pos_embed = self.pos_enc(
                batch_size, depot_num, lock_depot
            )  # (batch_size, depot_num, embed_dim)

            start_context_embed += pos_embed  # (batch_size, depot_num, embed_dim)
            end_context_embed += pos_embed  # (batch_size, depot_num, embed_dim)

        init_Z = torch.cat((start_context_embed, end_context_embed), dim=1)

        if lock_depot is None:
            mask_for_Z = None

            endpoint_mask = torch.zeros(
                (batch_size, depot_num * 2), dtype=torch.bool, device=graph_embed.device
            )
        else:
            # when one instance in a batch is unable to re-constructed, there will be all true in its lock_depot.
            mask_for_Z = lock_depot.repeat(1, 2)  # (batch_size, depot_num*2)
            mask_for_Z[:, 0][mask_for_Z.all(1)] = False

            endpoint_mask = mask_for_Z

        init_H = torch.cat(
            (start_context_embed, end_context_embed, available_embed), dim=1
        )

        if unavailable_mask is None:
            unavailable_mask = torch.zeros_like(
                available_index[:, depot_num:], dtype=torch.bool
            )

        cated_unavailable_mask_for_H = torch.cat(
            (endpoint_mask, unavailable_mask[:, depot_num:]), dim=1
        )

        final_Z: Tensor  # (batch_size, depot_num*2, embed_dim)
        final_H: Tensor  # (batch_size, depot_num*2+node_num, embed_dim)
        final_Z, final_H, *_ = self.encoding_blocks(
            (init_Z, init_H, mask_for_Z, cated_unavailable_mask_for_H)
        )

        final_Z_start = final_Z[:, :depot_num, :]
        final_Z_end = final_Z[:, depot_num:, :]

        combine_final_Z = torch.cat(
            (final_Z_start, final_Z_end), dim=2
        )  # (batch_size, depot_num, embed_dim*2)

        compatability = (
            torch.tanh(self.SHA_score(combine_final_Z, final_H[:, depot_num:, :])) * C
        ).squeeze(
            0
        )  # (batch_size, depot_num, depot_num+node_num)

        compatability[infeasible_mask] = -float('inf')
        prob = (
            compatability.view(batch_size, -1)
            .softmax(1)
            .view(batch_size, depot_num, depot_num + node_num)
        )

        return prob
