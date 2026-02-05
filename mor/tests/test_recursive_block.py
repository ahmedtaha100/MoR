import pytest

import torch



import sys

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))



from model.config import MoRConfig, get_mor_135m_config

from model.recursive_block import RecursiveBlock

from model.embeddings import MoREmbeddings

from model.attention import create_causal_mask





class TestRecursiveBlock:

    def get_inputs(self, config, batch=2, seq=64):

        embeddings = MoREmbeddings(config)



        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)



        hidden_states = embeddings(input_ids, position_ids)

        cos, sin = embeddings.get_rotary_emb(hidden_states, position_ids)

        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)



        return hidden_states, attention_mask, position_ids, cos, sin



    def test_output_shape(self):

        config = get_mor_135m_config()

        block = RecursiveBlock(config, layer_idx_start=1)

        block.eval()



        batch, seq = 2, 64

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        output, aux_outputs, aux_loss = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        assert output.shape == hidden_states.shape



    def test_routing_info_expert_choice(self):

        config = get_mor_135m_config()

        config.router_type = "expert_choice"

        block = RecursiveBlock(config, layer_idx_start=1)

        block.eval()



        batch, seq = 2, 64

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        _, aux_outputs, _ = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        routing_info = aux_outputs["routing_info"]

        assert "selected_masks" in routing_info

        assert "router_weights" in routing_info

        assert len(routing_info["selected_masks"]) == config.num_recursion_steps



    def test_routing_info_token_choice(self):

        config = get_mor_135m_config()

        config.router_type = "token_choice"

        block = RecursiveBlock(config, layer_idx_start=1)

        block.eval()



        batch, seq = 2, 64

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        _, aux_outputs, _ = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        routing_info = aux_outputs["routing_info"]

        assert "assigned_depth" in routing_info

        assert "depth_mask" in routing_info

        assert routing_info["assigned_depth"].shape == (batch, seq)



    def test_auxiliary_loss_training(self):

        config = get_mor_135m_config()

        block = RecursiveBlock(config, layer_idx_start=1)

        block.train()



        batch, seq = 2, 32

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        _, _, aux_loss = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        assert aux_loss is not None

        assert isinstance(aux_loss, torch.Tensor)



    def test_gradient_flow(self):

        config = get_mor_135m_config()

        block = RecursiveBlock(config, layer_idx_start=1)

        block.train()



        batch, seq = 2, 32

        hidden_states = torch.randn(batch, seq, config.hidden_dim, requires_grad=True)

        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)

        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)



        from model.embeddings import RotaryEmbedding

        rope = RotaryEmbedding(config.head_dim, config.max_seq_len)

        cos, sin = rope(hidden_states, position_ids)



        output, _, aux_loss = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        loss = output.sum()

        if aux_loss is not None and aux_loss.item() > 0:

            loss = loss + aux_loss

        loss.backward()



        assert hidden_states.grad is not None



    def test_recursion_depth_tracking(self):

        config = get_mor_135m_config()

        block = RecursiveBlock(config, layer_idx_start=1)

        block.eval()



        batch, seq = 2, 64

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        _, aux_outputs, _ = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        depths = block.get_recursion_depth_per_token(aux_outputs["routing_info"])



        assert depths.shape == (batch, seq)

        assert (depths >= 1).all()

        assert (depths <= config.num_recursion_steps).all()



    def test_hierarchical_filtering_expert_choice(self):

        config = get_mor_135m_config()

        config.router_type = "expert_choice"

        config.num_recursion_steps = 3

        block = RecursiveBlock(config, layer_idx_start=1)

        block.eval()



        batch, seq = 1, 100

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        _, aux_outputs, _ = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        selected_masks = aux_outputs["routing_info"]["selected_masks"]



        for r in range(1, len(selected_masks)):

            prev_mask = selected_masks[r - 1]

            curr_mask = selected_masks[r]

            assert (curr_mask & ~prev_mask).sum() == 0





class TestRecursiveBlockEdgeCases:

    def get_inputs(self, config, batch=2, seq=64):

        embeddings = MoREmbeddings(config)



        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)



        hidden_states = embeddings(input_ids, position_ids)

        cos, sin = embeddings.get_rotary_emb(hidden_states, position_ids)

        attention_mask = create_causal_mask(seq, hidden_states.device, hidden_states.dtype)



        return hidden_states, attention_mask, position_ids, cos, sin



    def test_short_sequence(self):

        config = get_mor_135m_config()

        block = RecursiveBlock(config, layer_idx_start=1)



        batch, seq = 1, 4

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        output, _, _ = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        assert output.shape == hidden_states.shape



    def test_single_recursion(self):

        config = get_mor_135m_config()

        config.num_recursion_steps = 1

        block = RecursiveBlock(config, layer_idx_start=1)



        batch, seq = 2, 32

        hidden_states, attention_mask, position_ids, cos, sin = self.get_inputs(config, batch, seq)



        output, aux_outputs, _ = block(

            hidden_states,

            attention_mask=attention_mask,

            position_ids=position_ids,

            cos=cos,

            sin=sin,

        )



        assert output.shape == hidden_states.shape





if __name__ == "__main__":

    pytest.main([__file__, "-v"])

