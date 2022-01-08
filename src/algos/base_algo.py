import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
from robosuite.utils.input_utils import input2action
from tqdm import tqdm

from models import Ensemble
from util import init_model


class BaseAlgorithm:
    def __init__(self, model, model_kwargs, save_dir, device, expert_policy=None, lr=1e-3, optimizer=torch.optim.Adam) -> None:
        self.model = model
        self.model_kwargs = model_kwargs
        self.save_dir = save_dir
        self.device = device
        self.expert_policy = expert_policy
        self.lr = lr

        self.model_type = type(model)
        self.optimizer_type = optimizer
        self.is_ensemble = self.model_type == Ensemble

        self._setup_optimizer()
        self._setup_metrics()

    def _setup_optimizer(self):
        if self.is_ensemble:
            self.optimizers = [
                self.optimizer_type(self.model.models[i].parameters(), lr=self.lr) for i in range(len(self.model.models))
            ]
        else:
            self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.lr)

    def _setup_metrics(self):
        if self.is_ensemble:
            self.ensemble_metrics = [defaultdict(list)] * len(self.model.models)
        else:
            self.metrics = defaultdict(list)

    def _reset_model(self):
        if self.is_ensemble:
            num_models = len(self.model.models)
            ensemble_model_type = type(self.model.models[0])
            model = init_model(ensemble_model_type, self.model_kwargs, device=self.device, num_models=num_models)
        else:
            model = init_model(self.model_type, self.model_kwargs, device=self.device, num_models=1)

        model.to(self.device)

    def _rollout(self, env, robosuite_cfg, trajectories_per_rollout, auto_only=False):
        data = []
        for j in range(trajectories_per_rollout):
            curr_obs, expert_mode = env.reset(), False
            done, success = False, False
            obs, act = [], []
            while not success and not done:
                obs.append(curr_obs.cpu())
                if expert_mode and not auto_only:
                    # Expert mode (either human or oracle algorithm)
                    a = self.expert_policy.act(curr_obs)
                    # act = np.clip(act, -act_limit, act_limit) TODO: clip actions?
                    if self._switch_mode(act=a) == True:
                        print("Switch to Robot")
                        expert_mode = False
                    next_obs, success, done, _ = env.step(a)
                else:
                    switch_mode = (not auto_only) and self._switch_mode(act=None, robosuite_cfg=robosuite_cfg, env=env)
                    if switch_mode:
                        print("Switch to Expert Mode")
                        expert_mode = True
                        continue
                    a = self.model(curr_obs).to(self.device)
                    next_obs, success, done, _ = env.step(a)
                act.append(a.cpu())
                curr_obs = next_obs
                
            demo = {"obs": obs, "act": act, "success": success}
            data.append(demo)
            env.close()

        return data

    def _save_checkpoint(self, epoch, best=False):
        if self.is_ensemble:
            # Save state dict for each model/optimizer
            ckpt_dict = {
                "models": [model.state_dict() for model in self.model.models],
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                "epoch": epoch,
            }
        else:
            ckpt_dict = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch}

        if best:
            ckpt_name = f"model_best.pt"
        else:
            ckpt_name = f"model_{epoch}.pt"

        save_path = os.path.join(self.save_dir, ckpt_name)
        torch.save(ckpt_dict, save_path)

    def _save_metrics(self):
        if self.is_ensemble:
            for i, metrics in enumerate(self.ensemble_metrics):
                save_path = os.path.join(self.save_dir, f"model{i}_metrics.pkl")
                df = pd.DataFrame(metrics)
                df.to_pickle(save_path)
        else:
            save_path = os.path.join(self.save_dir, "metrics.pkl")
            df = pd.DataFrame(self.metrics)
            df.to_pickle(save_path)

    def _update_metrics(self, **kwargs):
        if self.is_ensemble:
            for metrics in self.ensemble_metrics:
                for key, val in kwargs.items():
                    metrics[key].append(val)
        else:
            for key, val in kwargs.items():
                self.metrics[key].append(val)

    def train(self, model, optimizer, train_loader, val_loader, args):
        model.train()
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            prog_bar = tqdm(train_loader, leave=False)
            prog_bar.set_description(f"Epoch {epoch}/{args.epochs - 1}")
            epoch_losses = []
            for (obs, act) in prog_bar:
                optimizer.zero_grad()
                obs = obs.to(self.device)
                act = act.to(self.device)
                pred_act = model(obs)
                loss = torch.mean(torch.sum((act - pred_act) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                prog_bar.set_postfix(train_loss=loss.item())
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val_loss = self.validate(model, val_loader)

            # TODO wandb or tensorboard
            print(f"Epoch {epoch} Train Loss: {avg_loss}")
            print(f"Epoch {epoch} Val Loss: {avg_val_loss}")

            # Update metrics
            self._update_metrics(epoch=epoch, train_loss=avg_loss, val_loss=avg_val_loss)

            if epoch % args.save_iter == 0 or epoch == args.epochs - 1:
                self._save_checkpoint(epoch)
            
            if avg_val_loss < best_val_loss:
                self._save_checkpoint(epoch, best=True)
                best_val_loss = avg_val_loss

    def validate(self, model, val_loader):
        model.eval()
        val_losses = []
        for (obs, act) in val_loader:
            obs = obs.to(self.device)
            act = act.to(self.device)
            pred_act = model(obs)
            loss = torch.mean(torch.sum((act - pred_act) ** 2, dim=1))
            val_losses.append(loss.item())
        return sum(val_losses) / len(val_losses)

    def run(self, train_loader, val_loader, args, env=None, robosuite_cfg=None) -> None:
        raise NotImplementedError

    def eval_auto(self, args, env=None, robosuite_cfg=None):
        data = self._rollout(env, robosuite_cfg, args.N_eval_trajectories, auto_only=True)
        successes = [demo["success"] for demo in data]
        save_file = os.path.join(self.save_dir, "eval_auto_data.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Success rate: {sum(successes)/len(successes)}")
        print(f"Eval data saved to {save_file}")
