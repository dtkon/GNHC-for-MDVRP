from typing import Optional, Union
import torch
from torch import Tensor, nn
from deprecated import deprecated

from . import layers


class AM_Encoder(layers.Policy):
    def __init__(
        self,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        n_heads: int = 8,
        n_blocks_graph: int = 3,
        normalization: str = 'layer',
        problem: str = 'vrp',
    ) -> None:
        super().__init__()

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        if problem == 'vrp':
            self.graph_encoder: Union[layers.AM_Encoder_CVRP, layers.AM_Encoder_TSP] = (
                layers.AM_Encoder_CVRP(
                    embedding_dim,
                    feed_forward_hidden,
                    n_heads,
                    n_blocks_graph,
                    normalization,
                )
            )
        elif problem == 'tsp':
            self.graph_encoder = layers.AM_Encoder_TSP(
                embedding_dim,
                feed_forward_hidden,
                n_heads,
                n_blocks_graph,
                normalization,
            )
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(
        self, problems: Tensor, depot_num: int = 1
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d) or 2(x,y)]
        '''

        # encoder
        graph_embed = self.graph_encoder(
            problems, depot_num=depot_num
        )  # (batch_size, problem_size, embedding_dim)
        avg_graph_embed = graph_embed.mean(1)  # (batch_size, embedding_dim)

        proj_key, proj_val, proj_key_for_glimpse = self.precomputing(graph_embed)

        return graph_embed, avg_graph_embed, proj_key, proj_val, proj_key_for_glimpse


class AM_Decoder(layers.Policy):
    def __init__(
        self, n_heads: int = 8, embedding_dim: int = 128, problem: str = 'vrp'
    ) -> None:
        super().__init__()

        self.problem = problem

        if problem == 'tsp':
            self.decoder = layers.AM_Decoder(n_heads, embedding_dim, 3 * embedding_dim)
            self.v1f = nn.Parameter(
                torch.zeros(1, embedding_dim * 2)
            )  # v1 and vf in AM paper
        else:
            self.decoder = layers.AM_Decoder(
                n_heads, embedding_dim, 2 * embedding_dim + 1
            )

        self.init_parameters()

    def forward(
        self,
        graph_embed: Tensor,
        avg_graph_embed: Tensor,
        proj_key: Tensor,
        proj_val: Tensor,
        proj_key_for_glimpse: Tensor,
        current_node: Tensor,
        remain_capacity_or_tsp_end: Tensor,
        mask: Tensor,
    ) -> Tensor:
        '''
        current_node: (batch_size, 1)

        remain_capacity_or_tsp_end: (batch_size, 1)

        mask: (batch_size, problem_size)

        return: prob(batch_size, customer_num)
        '''
        batch_size, problem_size, _ = graph_embed.size()

        batch_arange = torch.arange(batch_size, device=graph_embed.device)

        if self.problem == 'tsp':
            if current_node.numel() > 0:
                current_node_embed = graph_embed[batch_arange, current_node.view(-1), :]
                end_node_embed = graph_embed[
                    batch_arange, remain_capacity_or_tsp_end.view(-1), :
                ]
                context = torch.cat(
                    [avg_graph_embed, current_node_embed, end_node_embed],
                    1,
                )
            else:
                context = torch.cat(
                    [avg_graph_embed, self.v1f.repeat(batch_size, 1)],
                    1,
                )
        else:
            current_node_embed = graph_embed[batch_arange, current_node.view(-1), :]
            context = torch.cat(
                [avg_graph_embed, current_node_embed, remain_capacity_or_tsp_end],
                1,
            )

        prob = self.decoder(
            context.unsqueeze(1), proj_key, proj_val, proj_key_for_glimpse, mask
        ).squeeze(1)

        return prob


class AM_Decoder_MDVRP(layers.Policy):
    def __init__(
        self, n_heads: int = 8, embedding_dim: int = 128, with_avg: bool = False
    ) -> None:
        super().__init__()

        self.with_avg = with_avg
        if with_avg:
            self.context_encoder = layers.MultiHeadSelfAttention(
                n_heads, 3 * embedding_dim + 1
            )
            self.decoder = layers.AM_Decoder(
                n_heads, embedding_dim, 3 * embedding_dim + 1
            )

        else:
            self.context_encoder = layers.MultiHeadSelfAttention(
                n_heads, 2 * embedding_dim + 1
            )
            self.decoder = layers.AM_Decoder(
                n_heads, embedding_dim, 2 * embedding_dim + 1
            )

        self.init_parameters()

    def forward(
        self,
        graph_embed: Tensor,
        avg_graph_embed: Tensor,
        proj_key: Tensor,
        proj_val: Tensor,
        proj_key_for_glimpse: Tensor,
        current_node: Tensor,
        depot_node: Tensor,
        remain_capacity: Tensor,
        capacity: Tensor,
        done: Optional[Tensor],
        lock_depot: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        '''
        current_node, depot_node: (batch_size, depot_num)

        remain_capacity, capacity: (batch_size, depot_num)

        lock_depot: (batch_size, depot_num)

        mask: (batch_size, depot_num, problem_size)

        return: (batch_size, customer_num)
        '''
        embded_dim = graph_embed.size(2)

        # batch_arange = torch.arange(batch_size, device=graph_embed.device)

        current_node_embed = graph_embed.gather(
            1, current_node.unsqueeze(2).expand(-1, -1, embded_dim)
        )  # (batch_size, depot_num, embed_dim)
        depot_embed = graph_embed.gather(
            1, depot_node.unsqueeze(2).expand(-1, -1, embded_dim)
        )  # (batch_size, depot_num, embed_dim)

        if self.with_avg:
            context = torch.cat(
                [
                    avg_graph_embed.unsqueeze(1).expand(-1, current_node.size(1), -1),
                    depot_embed,
                    current_node_embed,
                    (remain_capacity / capacity).unsqueeze(2),
                ],
                2,
            )  # (batch_size, depot_num, 3*embed_dim+1)
        else:
            context = torch.cat(
                [
                    depot_embed,
                    current_node_embed,
                    (remain_capacity / capacity).unsqueeze(2),
                ],
                2,
            )  # (batch_size, depot_num, 2*embed_dim+1)

        if lock_depot is not None:
            mask_for_context = lock_depot.clone()
            mask_for_context[:, 0][lock_depot.all(1)] = False

            update_context = self.context_encoder(context, mask=mask_for_context)
        else:
            update_context = self.context_encoder(context)

        # avoid all mask in one line to produce NaN in glimpse.
        # when one instance in a batch is done already, there will be only one false in its mask.
        mask_for_glimpse = mask.clone()
        mask_for_glimpse[:, :, 0][mask_for_glimpse.all(2)] = False

        prob = self.decoder(
            update_context,
            proj_key,
            proj_val,
            proj_key_for_glimpse,
            mask,
            mask_for_glimpse,
            cross_prob=True,
        )  # (batch_size, depot_num, problem_size)

        return prob


class AMre_Encoder(layers.Policy):
    def __init__(
        self,
        embedding_dim: int = 128,
        problem: str = 'vrp',
    ) -> None:
        super().__init__()

        if problem == 'vrp':
            self.graph_encoder: Union[layers.AM_Encoder_CVRP, layers.AM_Encoder_TSP] = (
                layers.AM_Encoder_CVRP(embedding_dim, 0, 0, 0, '')
            )
        elif problem == 'tsp':
            self.graph_encoder = layers.AM_Encoder_TSP(embedding_dim, 0, 0, 0, '')
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(self, problems: Tensor, depot_num: int = 1) -> tuple[Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d) or 2(x,y)]
        '''

        # encoder
        graph_init_embed = self.graph_encoder(
            problems, depot_num=depot_num
        )  # (batch_size, problem_size, embedding_dim)

        return (graph_init_embed,)


class AMre_Decoder(layers.Policy):
    def __init__(
        self,
        n_heads: int = 8,
        n_blocks: int = 3,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        normalization='instance',
        problem: str = 'vrp',
    ) -> None:
        super().__init__()

        self.problem = problem

        self.encoding_blocks: nn.Module = nn.Sequential(
            *(
                layers.EncodingBlock(
                    n_heads, embedding_dim, feed_forward_hidden, normalization
                )
                for _ in range(n_blocks)
            )
        )

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        if problem == 'tsp':
            self.decoder = layers.AM_Decoder(n_heads, embedding_dim, 3 * embedding_dim)
            self.v1f = nn.Parameter(
                torch.zeros(1, embedding_dim * 2)
            )  # v1 and vf in AM paper
        else:
            self.decoder = layers.AM_Decoder(
                n_heads, embedding_dim, 2 * embedding_dim + 1
            )

        self.init_parameters()

    def forward(
        self,
        graph_init_embed: Tensor,
        current_node: Tensor,
        available_index: Tensor,
        unavailable_mask: Optional[Tensor],
        remain_capacity_or_tsp_end: Tensor,
        infeasible_mask: Tensor,
    ) -> Tensor:
        '''
        current_node: (batch_size, 1)

        available_index: (batch_size, node_num)

        unavailable_mask: (batch_size, node_num), current_node should be available

        remain_capacity_or_tsp_end: (batch_size, 1)

        infeasible_mask: (batch_size, problem_size)

        return: prob(batch_size, customer_num)
        '''
        batch_size, problem_size, embedding_dim = graph_init_embed.size()

        available_embed = graph_init_embed.gather(
            1, available_index.unsqueeze(2).expand(-1, -1, embedding_dim)
        )

        # encoder
        graph_embed: Tensor
        graph_embed, _ = self.encoding_blocks(
            (available_embed, unavailable_mask)
        )  # (batch_size, problem_size, embedding_dim)

        if unavailable_mask is not None:
            masked_graph_embed = graph_embed.clone()
            masked_graph_embed[
                unavailable_mask.unsqueeze(2).expand(-1, -1, embedding_dim)
            ] = 0.0
            masked_avg_graph_embed = masked_graph_embed.sum(1) / (
                (~unavailable_mask).sum(1, keepdim=True)
                + 1e-6  # (batch_size, embedding_dim)
            )
        else:
            masked_avg_graph_embed = graph_embed.mean(1)  # (batch_size, embedding_dim)

        proj_key, proj_val, proj_key_for_glimpse = self.precomputing(graph_embed)

        batch_arange = torch.arange(batch_size, device=graph_init_embed.device)

        if self.problem == 'tsp':
            if current_node.numel() > 0:
                current_node_embed = graph_init_embed[
                    batch_arange, current_node.view(-1), :
                ]
                end_node_embed = graph_init_embed[
                    batch_arange, remain_capacity_or_tsp_end.view(-1), :
                ]
                context = torch.cat(
                    [masked_avg_graph_embed, current_node_embed, end_node_embed],
                    1,
                )
            else:
                context = torch.cat(
                    [masked_avg_graph_embed, self.v1f.repeat(batch_size, 1)],
                    1,
                )
        else:
            current_node_embed = graph_init_embed[
                batch_arange, current_node.view(-1), :
            ]
            context = torch.cat(
                [
                    masked_avg_graph_embed,
                    current_node_embed,
                    remain_capacity_or_tsp_end,
                ],
                1,
            )

        prob = self.decoder(
            context.unsqueeze(1),
            proj_key,
            proj_val,
            proj_key_for_glimpse,
            infeasible_mask,
        ).squeeze(1)

        return prob


@deprecated('bad to directly construct in actor')
class AM_Encoder_Construct(layers.Policy):
    def __init__(
        self,
        embedding_dim: int = 128,
        feed_forward_hidden: int = 512,
        n_heads: int = 8,
        n_blocks_graph: int = 3,
        normalization: str = 'layer',
        problem: str = 'vrp',
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim // n_heads

        if problem == 'vrp':
            self.graph_encoder: Union[layers.AM_Encoder_CVRP, layers.AM_Encoder_TSP] = (
                layers.AM_Encoder_CVRP(
                    embedding_dim,
                    feed_forward_hidden,
                    n_heads,
                    n_blocks_graph,
                    normalization,
                )
            )
        elif problem == 'tsp':
            self.graph_encoder = layers.AM_Encoder_TSP(
                embedding_dim,
                feed_forward_hidden,
                n_heads,
                n_blocks_graph,
                normalization,
            )
        else:
            raise NotImplementedError

        self.init_parameters()

    def forward(self, problems: Tensor) -> tuple[Tensor, Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d) or 2(x,y)]
        '''

        # encoder
        graph_embed = self.graph_encoder(
            problems
        )  # (batch_size, problem_size, embedding_dim)
        avg_graph_embed = graph_embed.mean(1)  # (batch_size, embedding_dim)

        return graph_embed, avg_graph_embed


@deprecated('bad to directly construct in actor')
class AM_Decoder_CVRP_Construct(layers.Policy):
    def __init__(self, n_heads: int = 8, embedding_dim: int = 128) -> None:
        super().__init__()

        self.precomputing = layers.AM_Decoder_Precompute(embedding_dim)

        self.decoder = layers.AM_Decoder_construct(
            n_heads, embedding_dim, 2 * embedding_dim + 1
        )

        self.init_parameters()

    def forward(
        self,
        problems: Tensor,
        graph_embed: Tensor,
        avg_graph_embed: Tensor,
        decoder_type: str = 'sample',
    ) -> tuple[Tensor, Tensor]:
        '''
        problems: [batch_size, problem_size, 3(x,y,d)]

        decoder_type: sample or greedy

        return: [actions(batch_size, node_indexes), log_prob(batch_size,)]
        '''
        batch_size, problem_size, _ = problems.size()

        batch_arange = torch.arange(batch_size, device=problems.device)

        proj_key, proj_val, proj_key_for_glimpse = self.precomputing(graph_embed)

        # track solution
        actions = torch.zeros((batch_size, 1), device=problems.device, dtype=torch.long)
        log_prob_list = []

        remain_capacity = torch.ones((batch_size, 1), device=problems.device)
        assign_count = torch.ones_like(actions)
        current_node_embed = graph_embed[:, 0, :]

        # prepare mask
        mask, selected_mask, ref_mask = AM_Decoder_CVRP_Construct.prepare_mask(
            batch_size, problem_size, problems.device
        )

        while True:
            ### prepare context
            context = torch.cat(
                [avg_graph_embed, current_node_embed, remain_capacity],
                1,
            )
            ###

            ### decoder forward
            next_node, log_prob = self.decoder(
                context.unsqueeze(1),
                proj_key,
                proj_val,
                proj_key_for_glimpse,
                mask,
                select_type=decoder_type,
            )
            next_node = next_node.view(-1)
            ###

            ### process return
            actions = torch.cat((actions, next_node.view(-1, 1)), dim=1)
            log_prob_list.append(log_prob.reshape(-1))
            ###

            ### if all done
            assign_count += (next_node != 0).view(-1, 1)
            if torch.all(assign_count == problem_size):
                actions = torch.cat(
                    (
                        actions,
                        torch.zeros(
                            (batch_size, 1), device=problems.device, dtype=torch.long
                        ),
                    ),
                    dim=1,
                )
                # self.decoder.clear_cache()
                break
            ###

            ### calculate new context
            remain_capacity -= problems[batch_arange, next_node, 2].view(batch_size, -1)
            remain_capacity[next_node == 0] = 1

            current_node_embed = graph_embed[batch_arange, next_node, :]
            ###

            ### update mask
            demand_too_large = (
                remain_capacity.expand(-1, problem_size) < problems[:, :, 2]
            )  # (batch_size, problem_size)

            done = (assign_count == problem_size).view(-1)  # (batch,)

            mask = AM_Decoder_CVRP_Construct.update_mask(
                batch_arange,
                next_node,
                selected_mask,
                ref_mask,
                demand_too_large,
                done,
            )
            ###

        log_probs = torch.stack(log_prob_list, 1).sum(1)  # (batch_size,)

        return actions, log_probs

    @staticmethod
    def prepare_mask(
        batch_size: int,
        problem_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        '''
        return: mask(used in decoder), selected_mask(used to be updated), done_mask(fixed for reference)
        '''
        selected_mask = torch.zeros(
            (batch_size, problem_size), device=device, dtype=torch.bool
        )
        selected_mask[:, 0] = True
        mask = selected_mask.clone()
        done_mask = torch.ones((1, problem_size), device=device, dtype=torch.bool)
        done_mask[:, 0] = False

        return mask, selected_mask, done_mask

    @staticmethod
    def update_mask(
        batch_arange: Tensor,
        next_node: Tensor,
        selected_mask: Tensor,
        done_mask: Tensor,
        demand_too_large: Tensor,
        done: Tensor,
    ) -> Tensor:
        '''
        selected_mask will be changed in-place
        '''
        selected_mask[batch_arange, next_node] = True  # (batch_size, problem_size)

        return_to_depot = next_node == 0  # (batch,)

        mask = selected_mask.clone()
        mask[demand_too_large] = True

        mask[:, 0] = False
        mask[:, 0][return_to_depot] = True

        mask[done, :] = done_mask

        return mask
