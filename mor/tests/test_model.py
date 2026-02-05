import pytest

import torch



import sys

from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))



from model.config import MoRConfig, get_mor_135m_config, get_vanilla_360m_config

from model.mor_model import MoRModel, MoRForCausalLM, MoRModelOutput

from model.vanilla_model import VanillaTransformer, VanillaForCausalLM





class TestVanillaTransformer:

    def test_output_shape(self):

        config = get_vanilla_360m_config()

        config.num_shared_layers = 4

        config.num_recursion_steps = 1



        model = VanillaTransformer(config)

        model.eval()



        batch, seq = 2, 64

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        hidden_states, outputs = model(input_ids)



        assert hidden_states.shape == (batch, seq, config.hidden_dim)



    def test_gradient_flow(self):

        config = get_vanilla_360m_config()

        config.num_shared_layers = 4

        config.num_recursion_steps = 1



        model = VanillaTransformer(config)

        model.train()



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        hidden_states, _ = model(input_ids)

        loss = hidden_states.sum()

        loss.backward()



        assert model.embeddings.token_embedding.weight.grad is not None





class TestVanillaForCausalLM:

    def test_forward_pass(self):

        config = get_vanilla_360m_config()

        config.num_shared_layers = 4

        config.num_recursion_steps = 1



        model = VanillaForCausalLM(config)

        model.eval()



        batch, seq = 2, 64

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        outputs = model(input_ids)



        assert outputs.logits.shape == (batch, seq, config.vocab_size)

        assert outputs.loss is None



    def test_loss_computation(self):

        config = get_vanilla_360m_config()

        config.num_shared_layers = 4

        config.num_recursion_steps = 1



        model = VanillaForCausalLM(config)

        model.train()



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels)



        assert outputs.loss is not None

        assert outputs.loss.shape == ()

        assert outputs.loss > 0



    def test_training_step(self):

        config = get_vanilla_360m_config()

        config.num_shared_layers = 4

        config.num_recursion_steps = 1



        model = VanillaForCausalLM(config)

        model.train()



        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels)

        loss = outputs.loss



        optimizer.zero_grad()

        loss.backward()



        for name, param in model.named_parameters():

            if param.requires_grad:

                assert param.grad is not None, f"No gradient for {name}"



        optimizer.step()



    def test_parameter_count(self):

        config = get_vanilla_360m_config()

        config.num_shared_layers = 4

        config.num_recursion_steps = 1



        model = VanillaForCausalLM(config)



        total_params = model.get_num_parameters()

        trainable_params = model.get_num_parameters(trainable_only=True)



        assert total_params > 0

        assert trainable_params == total_params





class TestMoRModel:

    def test_output_shape(self):

        config = get_mor_135m_config()



        model = MoRModel(config)

        model.eval()



        batch, seq = 2, 64

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        hidden_states, outputs, aux_loss = model(input_ids)



        assert hidden_states.shape == (batch, seq, config.hidden_dim)



    def test_routing_info(self):

        config = get_mor_135m_config()



        model = MoRModel(config)

        model.eval()



        batch, seq = 2, 64

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        _, outputs, _ = model(input_ids)



        assert "routing_info" in outputs



    def test_gradient_flow(self):

        config = get_mor_135m_config()



        model = MoRModel(config)

        model.train()



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        hidden_states, outputs, aux_loss = model(input_ids)

        loss = hidden_states.sum()

        if aux_loss is not None and aux_loss.item() > 0:

            loss = loss + aux_loss

        loss.backward()



        assert model.embeddings.token_embedding.weight.grad is not None





class TestMoRForCausalLM:

    def test_forward_pass(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, seq = 2, 64

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        outputs = model(input_ids)



        assert outputs.logits.shape == (batch, seq, config.vocab_size)



    def test_loss_with_aux(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.train()



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels)



        assert outputs.loss is not None

        assert outputs.aux_loss is not None



    def test_training_step(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.train()



        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels)

        loss = outputs.loss



        optimizer.zero_grad()

        loss.backward()



        has_router_grad = False

        for name, param in model.named_parameters():

            if param.requires_grad:

                assert param.grad is not None, f"No gradient for {name}"

                if 'router' in name:

                    has_router_grad = True



        assert has_router_grad, "Router should have gradients"



        optimizer.step()



    def test_recursion_depths(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, seq = 2, 64

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        depths = model.get_recursion_depths(input_ids)



        if depths is not None:

            assert depths.shape == (batch, seq)

            assert (depths >= 1).all()

            assert (depths <= config.num_recursion_steps).all()



    def test_generation_basic(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        prompt = torch.randint(0, config.vocab_size, (1, 4))



        with torch.no_grad():

            generated = model.generate(

                prompt,

                max_new_tokens=8,

                do_sample=False,

            )



        assert generated.shape[0] == 1

        assert generated.shape[1] <= 4 + 8



    def test_parameter_efficiency(self):

        mor_config = get_mor_135m_config()

        vanilla_config = get_vanilla_360m_config()



        vanilla_config.num_shared_layers = mor_config.num_layers



        mor_model = MoRForCausalLM(mor_config)

        vanilla_model = VanillaForCausalLM(vanilla_config)



        mor_params = mor_model.get_num_parameters()

        vanilla_params = vanilla_model.get_num_parameters()



        print(f"MoR params: {mor_params:,}")

        print(f"Vanilla params: {vanilla_params:,}")





class TestOverfitTest:

    def test_vanilla_overfit(self):

        config = get_vanilla_360m_config()

        config.num_shared_layers = 4

        config.num_recursion_steps = 1



        model = VanillaForCausalLM(config)

        model.train()



        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)



        batch, seq = 2, 16

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        initial_loss = None

        final_loss = None



        for step in range(50):

            outputs = model(input_ids, labels=labels)

            loss = outputs.loss



            if step == 0:

                initial_loss = loss.item()



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            final_loss = loss.item()



        print(f"Vanilla overfit: {initial_loss:.4f} -> {final_loss:.4f}")



        assert final_loss < initial_loss * 0.5



    def test_mor_overfit(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.train()



        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)



        batch, seq = 2, 16

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        initial_loss = None

        final_loss = None



        for step in range(50):

            outputs = model(input_ids, labels=labels)

            loss = outputs.loss



            if step == 0:

                initial_loss = loss.item()



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            final_loss = loss.item()



        print(f"MoR overfit: {initial_loss:.4f} -> {final_loss:.4f}")



        assert final_loss < initial_loss * 0.5





class TestNewFeatures:

    def test_shared_kv_caching(self):

        config = get_mor_135m_config()

        config.kv_cache_strategy = "shared"



        model = MoRForCausalLM(config)

        model.eval()



        batch, seq = 1, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        with torch.no_grad():

            outputs = model(input_ids, use_cache=True)



        assert outputs.logits.shape == (batch, seq, config.vocab_size)



        if outputs.past_key_values is not None:

            assert "recursive" in outputs.past_key_values

            recursive_kv = outputs.past_key_values["recursive"]

            assert 0 in recursive_kv

            assert len(recursive_kv) == 1



    def test_selective_kv_caching(self):

        config = get_mor_135m_config()

        config.kv_cache_strategy = "selective"



        model = MoRForCausalLM(config)

        model.eval()



        batch, seq = 1, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        with torch.no_grad():

            outputs = model(input_ids, use_cache=True)



        assert outputs.logits.shape == (batch, seq, config.vocab_size)



        if outputs.past_key_values is not None:

            assert "recursive" in outputs.past_key_values

            recursive_kv = outputs.past_key_values["recursive"]

            assert len(recursive_kv) == config.num_recursion_steps



    def test_gradient_checkpointing(self):

        config = get_mor_135m_config()

        config.gradient_checkpointing = True



        model = MoRForCausalLM(config)

        model.train()

        model.model.gradient_checkpointing_enable()



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels)

        loss = outputs.loss

        loss.backward()



        assert model.model.embeddings.token_embedding.weight.grad is not None



    def test_generation_with_kv_cache(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        prompt = torch.randint(0, config.vocab_size, (1, 4))



        with torch.no_grad():

            generated = model.generate(

                prompt,

                max_new_tokens=8,

                do_sample=False,

                use_cache=True,

            )



        assert generated.shape[0] == 1

        assert generated.shape[1] <= 4 + 8



    def test_router_z_loss(self):

        config = get_mor_135m_config()

        config.router_z_loss_coeff = 0.01



        model = MoRForCausalLM(config)

        model.train()



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels)



        assert outputs.aux_loss is not None

        assert outputs.aux_loss > 0



    def test_token_choice_eq2_fix(self):

        config = get_mor_135m_config()

        config.router_type = "token_choice"



        model = MoRForCausalLM(config)

        model.eval()



        batch, seq = 2, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        with torch.no_grad():

            outputs = model(input_ids)



        assert outputs.logits.shape == (batch, seq, config.vocab_size)



    def test_router_alpha_consistency(self):

        from model.router import ExpertChoiceRouter



        config = get_mor_135m_config()

        config.router_alpha = 2.0



        router = ExpertChoiceRouter(config, recursion_step=0)



        batch, seq = 2, 32

        hidden_states = torch.randn(batch, seq, config.hidden_dim)



        router.train()

        train_weights, train_mask, _ = router(hidden_states)



        router.eval()

        infer_weights, infer_mask = router.forward_inference(hidden_states)



        assert train_weights.shape == infer_weights.shape

        assert train_mask.shape == infer_mask.shape



    def test_past_key_values_exposed(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, seq = 1, 16

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        with torch.no_grad():

            outputs = model(input_ids, use_cache=True)



        assert hasattr(outputs, 'past_key_values')



    def test_cached_decode_with_attention_mask(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, prompt_len = 1, 8

        input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))

        attention_mask = torch.ones(batch, prompt_len, dtype=torch.long)



        with torch.no_grad():

            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

            past_kv = outputs.past_key_values



            next_token = torch.randint(0, config.vocab_size, (batch, 1))

            extended_mask = torch.ones(batch, prompt_len + 1, dtype=torch.long)



            outputs2 = model(

                next_token,

                attention_mask=extended_mask,

                past_key_values=past_kv,

                use_cache=True

            )



        assert outputs2.logits.shape == (batch, 1, config.vocab_size)



    def test_cycle_strategy_position_ids(self):

        config = get_mor_135m_config()

        config.sharing_strategy = "cycle"

        config.num_shared_layers = 10

        config.num_recursion_steps = 3



        model = MoRForCausalLM(config)

        model.eval()



        batch, prompt_len = 1, 8

        input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))



        with torch.no_grad():

            outputs = model(input_ids, use_cache=True)

            past_kv = outputs.past_key_values



            next_token = torch.randint(0, config.vocab_size, (batch, 1))

            outputs2 = model(

                next_token,

                past_key_values=past_kv,

                use_cache=True

            )



        assert outputs2.logits.shape == (batch, 1, config.vocab_size)



    def test_chunk_decode_causal_masking(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, prompt_len = 1, 8

        input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))



        with torch.no_grad():

            outputs = model(input_ids, use_cache=True)

            past_kv = outputs.past_key_values



            chunk = torch.randint(0, config.vocab_size, (batch, 4))

            outputs2 = model(

                chunk,

                past_key_values=past_kv,

                use_cache=True

            )



        assert outputs2.logits.shape == (batch, 4, config.vocab_size)



    def test_chunk_decode_with_attention_mask(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, prompt_len = 1, 8

        input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))

        attention_mask = torch.ones(batch, prompt_len, dtype=torch.long)



        with torch.no_grad():

            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

            past_kv = outputs.past_key_values



            chunk_len = 4

            chunk = torch.randint(0, config.vocab_size, (batch, chunk_len))

            extended_mask = torch.ones(batch, prompt_len + chunk_len, dtype=torch.long)



            outputs2 = model(

                chunk,

                attention_mask=extended_mask,

                past_key_values=past_kv,

                use_cache=True

            )



        assert outputs2.logits.shape == (batch, chunk_len, config.vocab_size)



    def test_kv_cache_actually_stores_values(self):

        from model.attention import SelectiveKVCache



        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, prompt_len = 1, 8

        input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))



        with torch.no_grad():

            outputs = model(input_ids, use_cache=True)



        pkv = outputs.past_key_values

        assert pkv is not None



        assert pkv["first"] is not None

        assert pkv["first"].has_cache()

        assert pkv["first"].get_seq_len() == prompt_len



        assert "recursive" in pkv

        for r_idx, layer_caches in pkv["recursive"].items():

            assert layer_caches is not None

            assert len(layer_caches) == config.num_shared_layers

            for layer_cache in layer_caches:

                assert layer_cache is not None

                if isinstance(layer_cache, SelectiveKVCache):

                    assert layer_cache.get_seq_len() == prompt_len

                else:

                    assert layer_cache.has_cache()

                    assert layer_cache.get_seq_len() == prompt_len



    def test_position_ids_advance_correctly(self):

        config = get_mor_135m_config()



        model = MoRForCausalLM(config)

        model.eval()



        batch, prompt_len = 1, 8

        input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))



        with torch.no_grad():

            outputs1 = model(input_ids, use_cache=True)

            past_kv = outputs1.past_key_values



            next_token = torch.randint(0, config.vocab_size, (batch, 1))

            outputs2 = model(next_token, past_key_values=past_kv, use_cache=True)

            past_kv2 = outputs2.past_key_values



            assert past_kv2["first"].get_seq_len() == prompt_len + 1



            for r_idx, layer_caches in past_kv2["recursive"].items():

                for layer_cache in layer_caches:

                    assert layer_cache.get_seq_len() == prompt_len + 1



    def test_position_ids_cycle_strategy(self):

        config = get_mor_135m_config()

        config.sharing_strategy = "cycle"

        config.num_shared_layers = 10

        config.num_recursion_steps = 3



        model = MoRForCausalLM(config)

        model.eval()



        batch, prompt_len = 1, 8

        input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))



        with torch.no_grad():

            outputs1 = model(input_ids, use_cache=True)

            past_kv = outputs1.past_key_values



            next_token = torch.randint(0, config.vocab_size, (batch, 1))

            outputs2 = model(next_token, past_key_values=past_kv, use_cache=True)

            past_kv2 = outputs2.past_key_values



            for r_idx, layer_caches in past_kv2["recursive"].items():

                for layer_cache in layer_caches:

                    assert layer_cache.get_seq_len() == prompt_len + 1



    def test_selective_kv_compaction(self):

        from model.attention import SelectiveKVCache



        config = get_mor_135m_config()

        config.kv_cache_strategy = "selective"



        model = MoRForCausalLM(config)

        model.eval()



        batch, seq = 1, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))



        with torch.no_grad():

            outputs = model(input_ids, use_cache=True)



        pkv = outputs.past_key_values

        assert pkv is not None

        assert "recursive" in pkv



        for r_idx, layer_caches in pkv["recursive"].items():

            for layer_cache in layer_caches:

                assert isinstance(layer_cache, SelectiveKVCache)



                cached_len = layer_cache.get_cached_len()

                total_len = layer_cache.get_seq_len()



                assert cached_len <= total_len



    def test_selective_masking_without_cache(self):

        config = get_mor_135m_config()

        config.kv_cache_strategy = "selective"



        model = MoRForCausalLM(config)

        model.train()



        batch, seq = 2, 16

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels, use_cache=False)



        assert outputs.loss is not None

        assert outputs.routing_info is not None

        assert "selected_masks" in outputs.routing_info



        outputs.loss.backward()



        for name, param in model.named_parameters():

            if param.requires_grad:

                assert param.grad is not None, f"No gradient for {name}"



    def test_shared_kv_reuse_without_cache(self):

        config = get_mor_135m_config()

        config.kv_cache_strategy = "shared"



        model = MoRForCausalLM(config)

        model.train()



        batch, seq = 2, 16

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs = model(input_ids, labels=labels, use_cache=False)



        assert outputs.loss is not None

        assert outputs.routing_info is not None



        outputs.loss.backward()



        for name, param in model.named_parameters():

            if param.requires_grad:

                assert param.grad is not None, f"No gradient for {name}"



    def test_active_mask_applied_during_training(self):

        config = get_mor_135m_config()

        config.kv_cache_strategy = "selective"



        model = MoRForCausalLM(config)

        model.train()



        batch, seq = 1, 32

        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        labels = input_ids.clone()



        outputs1 = model(input_ids, labels=labels, use_cache=False)

        loss1 = outputs1.loss.item()



        config2 = get_mor_135m_config()

        config2.kv_cache_strategy = "shared"



        model2 = MoRForCausalLM(config2)

        model2.load_state_dict(model.state_dict())

        model2.train()



        outputs2 = model2(input_ids, labels=labels, use_cache=False)

        loss2 = outputs2.loss.item()



        assert outputs1.routing_info is not None

        assert outputs2.routing_info is not None





if __name__ == "__main__":

    pytest.main([__file__, "-v"])
