import torch
import torch.nn as nn
import torch.optim as optim

from models.model import ST_DTGCN
import utils

try:
    from torchinfo import summary
except ImportError:
    summary = None

try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None


class trainer(nn.Module):
    def __init__(
        self,
        scaler,
        adj,
        history,
        num_of_vertices,
        in_dim,
        hidden_dims,
        first_layer_embedding_size,
        out_layer_dim,
        d_model,
        n_heads,
        factor,
        attention_dropout,
        output_attention,
        dropout,
        forward_expansion,
        log,
        lrate,
        w_decay,
        l_decay_rate,
        device,
        activation='GLU',
        use_mask=True,
        max_grad_norm=5,
        lr_decay=False,
        temporal_emb=True,
        spatial_emb=True,
        use_transformer=True,
        use_informer=True,
        use_adaptive_graph=True,
        horizon=12,
        strides=12,
        loss_function='mae',
        huber_delta=1.0,
        norm_type="LayerNorm",
        ffn_activation="relu",
        return_stability_stats=False,
        use_temporal_inception = False
    ):
        super(trainer, self).__init__()

        self.device = device
        self.scaler = scaler
        self.clip = max_grad_norm
        self.use_mask = use_mask
        self.history = history
        self.num_of_vertices = num_of_vertices
        self.in_dim = in_dim
        self.horizon = horizon
        self.huber_delta = huber_delta
        self.loss_name = loss_function.lower()

        self.model = ST_DTGCN(
            adj=adj,
            history=history,
            num_of_vertices=num_of_vertices,
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            first_layer_embedding_size=first_layer_embedding_size,
            out_layer_dim=out_layer_dim,
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=attention_dropout,
            output_attention=output_attention,
            dropout=dropout,
            forward_expansion=forward_expansion,
            horizon=horizon,
            strides=strides,
            activation=activation,
            temporal_emb=temporal_emb,
            spatial_emb=spatial_emb,
            use_transformer=use_transformer,
            use_informer=use_informer,
            use_adaptive_graph=use_adaptive_graph,
            norm_type=norm_type,
            ffn_activation=ffn_activation,
            return_stability_stats=return_stability_stats,
            use_temporal_inception=use_temporal_inception, 
        )

        self.model = self.model.to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lrate,
            weight_decay=w_decay
        )

        self.lr_scheduler = None
        if lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: l_decay_rate ** epoch
            )

        if self.loss_name == "mae":
            self.loss = lambda pred, real, null_val=0.0: utils.masked_mae(
                pred, real, null_val=null_val
            )
        elif self.loss_name == "mse":
            self.loss = lambda pred, real, null_val=0.0: utils.masked_mse(
                pred, real, null_val=null_val
            )
        elif self.loss_name == "huber":
            self.loss = lambda pred, real, null_val=0.0: utils.masked_huber(
                pred,
                real,
                delta=self.huber_delta,
                null_val=null_val
            )
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")
    @staticmethod
    def _format_num(num):
        if num >= 1e9:
            return f"{num / 1e9:.3f}B"
        if num >= 1e6:
            return f"{num / 1e6:.3f}M"
        if num >= 1e3:
            return f"{num / 1e3:.3f}K"
        return str(num)

    def _base_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def get_param_summary(self):
        model = self._base_model()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
        }

    def get_layer_param_details(self):
        model = self._base_model()
        details = []

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if params > 0:
                    details.append({
                        "name": name,
                        "type": module.__class__.__name__,
                        "params": params,
                        "trainable": trainable,
                    })
        return details

    def print_model_structure(self, log=None):
        model = self._base_model()
        text = str(model)
        if log is not None:
            utils.log_string(log, "\n[Model Structure]")
            utils.log_string(log, text)
        else:
            print("\n[Model Structure]")
            print(text)

    def print_param_summary(self, log=None):
        summary_dict = self.get_param_summary()

        lines = [
            "\n[Parameter Summary]",
            f"Total Parameters     : {summary_dict['total_params']:,} ({self._format_num(summary_dict['total_params'])})",
            f"Trainable Parameters : {summary_dict['trainable_params']:,} ({self._format_num(summary_dict['trainable_params'])})",
            f"Non-trainable Params : {summary_dict['non_trainable_params']:,} ({self._format_num(summary_dict['non_trainable_params'])})",
        ]

        if log is not None:
            for line in lines:
                utils.log_string(log, line)
        else:
            for line in lines:
                print(line)

    def print_layer_params(self, log=None):
        details = self.get_layer_param_details()

        if log is not None:
            utils.log_string(log, "\n[Per-layer Parameter Statistics]")
            for item in details:
                utils.log_string(
                    log,
                    f"{item['name']:60s} | {item['type']:25s} | "
                    f"params: {item['params']:10,d} | trainable: {item['trainable']:10,d}"
                )
        else:
            print("\n[Per-layer Parameter Statistics]")
            for item in details:
                print(
                    f"{item['name']:60s} | {item['type']:25s} | "
                    f"params: {item['params']:10,d} | trainable: {item['trainable']:10,d}"
                )

    def print_torchinfo_summary(self, log=None, batch_size=1):
        model = self._base_model()

        if summary is None:
            msg = "torchinfo summary skipped (torchinfo not installed). Install: pip install torchinfo"
            if log is not None:
                utils.log_string(log, msg)
            else:
                print(msg)
            return

        try:
            info = summary(
                model,
                input_size=(batch_size, self.history, self.num_of_vertices, self.in_dim),
                device=str(self.device),
                verbose=0,
                col_names=("input_size", "output_size", "num_params", "trainable"),
                depth=5,
            )
            if log is not None:
                utils.log_string(log, "\n[torchinfo Summary]")
                utils.log_string(log, str(info))
            else:
                print("\n[torchinfo Summary]")
                print(info)
        except Exception as e:
            msg = f"torchinfo summary failed: {e}"
            if log is not None:
                utils.log_string(log, msg)
            else:
                print(msg)

    def print_flops(self, log=None, batch_size=1):
        model = self._base_model()

        if profile is None or clever_format is None:
            msg = "FLOPs/MACs skipped (thop not installed). Install: pip install thop"
            if log is not None:
                utils.log_string(log, msg)
            else:
                print(msg)
            return

        try:
            dummy_input = torch.randn(
                batch_size, self.history, self.num_of_vertices, self.in_dim,
                device=self.device
            )
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            macs_str, params_str = clever_format([macs, params], "%.3f")

            lines = [
                "\n[FLOPs / MACs]",
                f"MACs          : {macs_str}",
                f"Params        : {params_str}",
                f"Approx FLOPs  : ~ {macs * 2:.3e}",
            ]

            if log is not None:
                for line in lines:
                    utils.log_string(log, line)
            else:
                for line in lines:
                    print(line)

        except Exception as e:
            msg = f"FLOPs/MACs profiling failed: {e}"
            if log is not None:
                utils.log_string(log, msg)
            else:
                print(msg)

    def print_full_report(self, log=None, batch_size=1, print_flops=True):
        header = "\n" + "=" * 90 + "\nMODEL REPORT\n" + "=" * 90
        if log is not None:
            utils.log_string(log, header)
        else:
            print(header)

        self.print_model_structure(log=log)
        self.print_param_summary(log=log)
        self.print_layer_params(log=log)
        self.print_torchinfo_summary(log=log, batch_size=batch_size)

        if print_flops:
            self.print_flops(log=log, batch_size=batch_size)

        footer = "=" * 90 + "\n"
        if log is not None:
            utils.log_string(log, footer)
        else:
            print(footer)

    def _inverse_transform(self, output):
        return self.scaler.inverse_transform(output)

    def _compute_metrics(self, predict, real_val):
        loss = self.loss(predict, real_val, 0.0)
        mae, rmse = utils.metric(predict, real_val)
        return loss, mae, rmse

    def train_model(self, input_data, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input_data)
        predict = self._inverse_transform(output)

        loss, mae, rmse = self._compute_metrics(predict, real_val)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        return loss.item(), mae, rmse

    def eval_model(self, input_data, real_val):
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_data)
            predict = self._inverse_transform(output)
            loss, mae, rmse = self._compute_metrics(predict, real_val)

        return loss.item(), mae, rmse