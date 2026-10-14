"""Small CPU example and execution checks; no data or checkpoints needed."""
import torch

from models.tokenizer import AutoEncoder, AutoEncoderConfig
from models.dynamics_model import DynamicsModel, DynamicsModelConfig
from training.train_tokenizer import normalize_loss


def check_loss_normalization():
    # Positive rescaling of a loss must not change its normalized gradient.
    # An attached denominator would instead cancel the gradient entirely.
    for scale in (0.001, 1., 1000.):
        x = torch.tensor([2., 3.], requires_grad=True)
        loss = 0.8 * normalize_loss(x[0].square()) + 0.2 * normalize_loss(scale * x[1].square())
        loss.backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.8, 0.4 / 3]))
    # The epsilon floor keeps exact and near-zero losses/gradients finite.
    for value in (0., 1e-12):
        loss = torch.tensor(value, requires_grad=True)
        normalized = normalize_loss(loss)
        normalized.backward()
        torch.testing.assert_close(normalized, torch.tensor(value / 1e-8))
        torch.testing.assert_close(loss.grad, torch.tensor(1e8))
    print('Passed: detached loss normalization, scale-invariant gradients, and zero-loss handling.')


def main():
    check_loss_normalization()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    shared = dict(
        dtype=torch.float32, embedding_dim=32, n_heads=4, depth=4,
        n_latents=4, bottleneck_dim=8, max_temporal_length=4,
        drop_rate=0.0, att_drop_rate=0.0,
    )
    tokenizer = AutoEncoder(AutoEncoderConfig(
        **shared, img_input_H=16, img_input_W=16,
    ))
    frames = torch.rand(2, 4, 16, 16, 3)
    reconstruction = tokenizer(frames)
    assert reconstruction.shape == frames.shape
    reconstruction_loss = (reconstruction - frames).square().mean()
    reconstruction_loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in tokenizer.parameters()
               if p.grad is not None)

    tokenizer.eval().requires_grad_(False)
    latents = tokenizer.encoder(frames)
    assert latents.shape == (2, 4, 4, 8)
    changed_frames = frames.clone()
    changed_frames[:, 2:] = torch.rand_like(changed_frames[:, 2:])
    torch.testing.assert_close(
        latents[:, :2], tokenizer.encoder(changed_frames)[:, :2],
    )
    changed_latents = latents.clone()
    changed_latents[:, 2:] = torch.randn_like(changed_latents[:, 2:])
    torch.testing.assert_close(
        tokenizer.decoder(latents)[:, :2],
        tokenizer.decoder(changed_latents)[:, :2],
    )

    for n_actions in (0, 3):
        dynamics = DynamicsModel(DynamicsModelConfig(**shared, n_actions=n_actions))
        actions = torch.randint(n_actions, (2, 10)) if n_actions else None
        clip_actions = actions[:, :4] if actions is not None else None
        loss = dynamics.loss(latents, action_idx=clip_actions)
        assert torch.isfinite(loss)
        loss.backward()
        assert dynamics.in_proj.weight.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in dynamics.parameters()
                   if p.grad is not None)

        dynamics.eval()
        with torch.no_grad():
            tau = dynamics.sample_tau(2, 4, 'cpu')
            features = dynamics.action_features(clip_actions)
            torch.testing.assert_close(
                dynamics(latents, tau, features)[:, :2],
                dynamics(changed_latents, tau, features)[:, :2],
            )
            future = dynamics.generate(latents, n_generate=6, K=4, action_idx=actions)
            assert future.shape == (2, 6, 4, 8)
            assert torch.isfinite(future).all()
            decoded = tokenizer.decoder(future[:, -4:])
            assert decoded.shape == frames.shape
            assert torch.isfinite(decoded).all()
        print(f'n_actions={n_actions}: loss={loss.item():.4f}, rollout={tuple(future.shape)}')

    print('Passed: reconstruction, gradients, causality, sliding rollout, and decoding.')


if __name__ == '__main__':
    main()
