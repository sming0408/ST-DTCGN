import os
import time
import csv
import argparse
import configparser
import ast
import random

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from engine import trainer
from utils import *

try:
    from torchinfo import summary
except ImportError:
    summary = None

try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y', 't'):
        return True
    if v.lower() in ('false', '0', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='ST_ITGCN Training Arguments', add_help=True)
parser.add_argument('--dataset', type=str, default='COVID_KOR', help='Dataset name')

# 输出开关
parser.add_argument('--print_model', action='store_true', help='Print model structure')
parser.add_argument('--print_params', action='store_true', help='Print parameter count')
parser.add_argument('--print_flops', action='store_true', help='Print FLOPs / MACs')

temp_args, _ = parser.parse_known_args()

config_file = f'/A.I_DATA/jbnu/Samyung/1ST_ITGCN/ST-DTGCN/config/{temp_args.dataset}.conf'
config = configparser.ConfigParser()
config.read(config_file, encoding='utf-8')

parser.add_argument('--no_cuda', action="store_true", help="Disable GPU")
parser.add_argument('--data', type=str, default=config.get('data', 'data', fallback=''), help='Data path')
parser.add_argument('--sensors_distance', type=str, default=config.get('data', 'sensors_distance', fallback=''), help='Node distance file')
parser.add_argument('--batch_size', type=int, default=config.getint('data', 'batch_size'), help="Training batch size")
parser.add_argument('--valid_batch_size', type=int, default=config.getint('data', 'valid_batch_size'), help="Validation batch size")
parser.add_argument('--test_batch_size', type=int, default=config.getint('test', 'test_batch_size', fallback=config.getint('data', 'valid_batch_size')), help="Test batch size")
parser.add_argument('--fill_zeros', type=str2bool, default=config.getboolean('data', 'fill_zeros'), help="Whether to fill zeros in data with average")
parser.add_argument('--num_of_vertices', type=int, default=config.getint('model', 'num_of_vertices'), help='Number of nodes')
parser.add_argument('--in_dim', type=int, default=config.getint('model', 'in_dim'), help='Input dimension')
parser.add_argument('--hidden_dims', type=list, default=ast.literal_eval(config.get('model', 'hidden_dims')), help='Hidden dimensions of each ST_ITGCNL layer')
parser.add_argument('--first_layer_embedding_size', type=int, default=config.getint('model', 'first_layer_embedding_size'), help='First layer embedding size')
parser.add_argument('--out_layer_dim', type=int, default=config.getint('model', 'out_layer_dim'), help='Output layer hidden dimension')
parser.add_argument('--d_model', type=int, default=config.getint('model', 'd_model'), help='Transformer embedding dimension')
parser.add_argument('--n_heads', type=int, default=config.getint('model', 'n_heads'), help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=config.getfloat('model', 'dropout'), help='Dropout rate')
parser.add_argument('--forward_expansion', type=int, default=config.getint('model', 'forward_expansion'), help='Transformer FFN expansion factor')
parser.add_argument('--use_adaptive_graph', type=str2bool, default=config.getboolean('model', 'use_adaptive_graph'), help="Whether to use adaptive unified graph")
parser.add_argument('--history', type=int, default=config.getint('model', 'history'), help="Input sequence length")
parser.add_argument('--horizon', type=int, default=config.getint('model', 'horizon'), help="Prediction horizon")
parser.add_argument('--strides', type=int, default=config.getint('model', 'strides'), help="Sliding window stride")
parser.add_argument('--temporal_emb', type=str2bool, default=config.getboolean('model', 'temporal_emb'), help="Use temporal embedding")
parser.add_argument('--spatial_emb', type=str2bool, default=config.getboolean('model', 'spatial_emb'), help="Use spatial embedding")
parser.add_argument('--use_transformer', type=str2bool, default=config.getboolean('model', 'use_transformer'), help="Use Transformer block")
parser.add_argument('--use_informer', type=str2bool, default=config.getboolean('model', 'use_informer'), help="Use Informer-style attention")
parser.add_argument('--factor', type=int, default=config.getint('model', 'factor'), help="ProbSparse factor")
parser.add_argument('--attention_dropout', type=float, default=config.getfloat('model', 'attention_dropout'), help="Attention dropout")
parser.add_argument('--output_attention', type=str2bool, default=config.getboolean('model', 'output_attention'), help="Output attention weights")
parser.add_argument('--use_mask', type=str2bool, default=config.getboolean('model', 'use_mask'), help="Whether to use mask matrix")
parser.add_argument('--activation', type=str, default=config.get('model', 'activation'), help="Activation function")
parser.add_argument('--seed', type=int, default=config.getint('train', 'seed'), help='Random seed')
parser.add_argument('--learning_rate', type=float, default=config.getfloat('train', 'learning_rate'), help="Initial learning rate")
parser.add_argument('--weight_decay', type=float, default=config.getfloat('train', 'weight_decay'), help="Weight decay")
parser.add_argument('--lr_decay', type=str2bool, default=config.getboolean('train', 'lr_decay'), help="Use lr decay")
parser.add_argument('--lr_decay_rate', type=float, default=config.getfloat('train', 'lr_decay_rate'), help="Learning rate decay rate")
parser.add_argument('--epochs', type=int, default=config.getint('train', 'epochs', fallback=100), help="Number of training epochs")
parser.add_argument('--print_every', type=int, default=config.getint('train', 'print_every', fallback=1), help='Print every N epochs')
parser.add_argument('--save', type=str, default=config.get('train', 'save', fallback='./checkpoints/'), help='Model save directory')
parser.add_argument('--save_loss', type=str, default=config.get('train', 'save_loss', fallback='./loss/'), help='Loss save directory')
parser.add_argument('--expid', type=int, default=config.getint('train', 'expid', fallback=1), help='Experiment id')
parser.add_argument('--max_grad_norm', type=float, default=config.getfloat('train', 'max_grad_norm', fallback=5.0), help="Gradient clipping threshold")
parser.add_argument('--patience', type=int, default=config.getint('train', 'patience', fallback=10), help='Early stopping patience')
parser.add_argument('--log_file', type=str, default=config.get('train', 'log_file', fallback='./train.log'), help='Log file path')
parser.add_argument('--loss_function', type=str, default=config.get('train', 'loss_function', fallback='mae'), help='Loss function: mae | mse | huber')
parser.add_argument('--huber_delta', type=float, default=config.getfloat('train', 'huber_delta', fallback=1.0), help='Huber delta')
parser.add_argument('--norm_type', type=str, default=config.get("model", "norm_type", fallback="DyTanhNorm"), help='Normalization type')
parser.add_argument('--ffn_activation', type=str, default=config.get("model", "ffn_activation", fallback="relu"), help='FFN activation')
parser.add_argument('--return_stability_stats', type=str2bool, default=config.getboolean("model", "return_stability_stats", fallback=False), help='Whether to return stability statistics')
parser.add_argument('--use_temporal_inception', type=str2bool, default=config.getboolean("model", "use_temporal_inception", fallback=True), help='Whether to use Temporal Inception module')

args = parser.parse_args()


def extract_data_suffix(data_path: str):
    name = os.path.basename(data_path).lower()
    if "confirmed" in name:
        return "_Confirmed"
    elif "deaths" in name:
        return "_Deaths"
    elif "recovered" in name:
        return "_Recovered"
    else:
        return "_Unknown"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_log_file_path(path: str, dataset_name: str, expid: int) -> str:
    if path.endswith(os.sep) or (os.path.exists(path) and os.path.isdir(path)):
        ensure_dir(path)
        return os.path.join(path, f'train_{dataset_name}_exp{expid}.log')

    root, ext = os.path.splitext(path)
    if ext == '':
        ensure_dir(path)
        return os.path.join(path, f'train_{dataset_name}_exp{expid}.log')

    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    return path


def get_best_model_path(save_dir: str, expid: int) -> str:
    ensure_dir(save_dir)
    return os.path.join(save_dir, f'exp_{expid}_best_model_by_val_loss.pth')


def get_loss_curve_path(save_loss_dir: str, expid: int) -> str:
    ensure_dir(save_loss_dir)
    return os.path.join(save_loss_dir, f'loss_curve_exp_{expid}.png')


def get_history_npz_path(save_loss_dir: str, expid: int) -> str:
    ensure_dir(save_loss_dir)
    return os.path.join(save_loss_dir, f'history_metrics_exp_{expid}.npz')


def get_result_root_dir(dataset_name: str, suffix: str) -> str:
    return os.path.join('/A.I_DATA/jbnu/Samyung/1ST_ITGCN/0326revision/results', dataset_name, suffix)


def get_results_csv_path(save_dir: str, dataset_name: str, suffix: str) -> str:
    ensure_dir(save_dir)
    clean_suffix = suffix.lstrip('_')
    return os.path.join(save_dir, f'{dataset_name}_{clean_suffix}_results.csv')


def get_epoch_memory_csv_path(save_dir: str, dataset_name: str, expid: int, suffix: str) -> str:
    ensure_dir(save_dir)
    clean_suffix = suffix.lstrip('_')
    return os.path.join(save_dir, f'{dataset_name}_{clean_suffix}_exp{expid}_memory_epoch.csv')


def get_memory_plot_path(save_dir: str, dataset_name: str, expid: int, suffix: str) -> str:
    ensure_dir(save_dir)
    clean_suffix = suffix.lstrip('_')
    return os.path.join(save_dir, f'{dataset_name}_{clean_suffix}_exp{expid}_memory_curve.png')


def get_per_horizon_csv_path(save_dir: str, dataset_name: str, expid: int, suffix: str) -> str:
    ensure_dir(save_dir)
    clean_suffix = suffix.lstrip('_')
    return os.path.join(save_dir, f'{dataset_name}_{clean_suffix}_exp{expid}_per_horizon_test.csv')


def get_test_debug_npz_path(save_dir: str, dataset_name: str, expid: int, suffix: str) -> str:
    ensure_dir(save_dir)
    clean_suffix = suffix.lstrip('_')
    return os.path.join(save_dir, f'{dataset_name}_{clean_suffix}_exp{expid}_test_debug_arrays.npz')


def log_file_only(log, string):
    if log is None:
        return
    log.write(str(string) + '\n')
    log.flush()


def print_and_log(log, string=""):
    print(string)
    log_file_only(log, string)


def format_num(num):
    if num >= 1e9:
        return f"{num / 1e9:.3f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.3f}K"
    return str(num)


def format_flops_value(num):
    if num is None:
        return "N/A"
    if num >= 1e9:
        return f"{num / 1e9:.4f} B"
    elif num >= 1e6:
        return f"{num / 1e6:.4f} M"
    elif num >= 1e3:
        return f"{num / 1e3:.4f} K"
    return f"{num:.4f}"


def reset_gpu_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_stats():
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0
        }
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 ** 2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 ** 2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2
    }


def save_epoch_memory_csv(csv_path, epoch_memory_records):
    if len(epoch_memory_records) == 0:
        return
    fieldnames = list(epoch_memory_records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_memory_records)


def plot_memory_vs_epoch(epoch_memory_records, save_path, log=None):
    if len(epoch_memory_records) == 0:
        return

    epochs = [r["epoch"] for r in epoch_memory_records]
    alloc = [r["allocated_mb"] for r in epoch_memory_records]
    reserved = [r["reserved_mb"] for r in epoch_memory_records]
    max_alloc = [r["max_allocated_mb"] for r in epoch_memory_records]
    max_reserved = [r["max_reserved_mb"] for r in epoch_memory_records]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, alloc, label='Allocated MB')
    plt.plot(epochs, reserved, label='Reserved MB')
    plt.plot(epochs, max_alloc, label='Max Allocated MB')
    plt.plot(epochs, max_reserved, label='Max Reserved MB')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.title('GPU Memory vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_file_only(log, f'Memory curve saved to: {save_path}')


def append_results_csv(csv_path, row_dict):
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def save_per_horizon_csv(csv_path, rows):
    if not rows:
        return
    ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_model_statistics_dict(model, input_shape, device):
    stats = {
        "total_params": None,
        "trainable_params": None,
        "non_trainable_params": None,
        "macs": None,
        "flops": None,
        "thop_params": None
    }

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    stats["total_params"] = total_params
    stats["trainable_params"] = trainable_params
    stats["non_trainable_params"] = non_trainable_params

    if profile is not None:
        try:
            dummy_input = torch.randn(*input_shape).to(device)
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
            stats["macs"] = float(macs)
            stats["flops"] = float(macs * 2)
            stats["thop_params"] = float(params)
        except Exception:
            pass

    return stats


def print_model_statistics(model, input_shape, device,
                           print_model=False, print_params=False, print_flops=False, log=None):
    stats = {
        "total_params": None,
        "trainable_params": None,
        "non_trainable_params": None,
        "macs": None,
        "flops": None,
        "thop_params": None
    }

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    stats["total_params"] = total_params
    stats["trainable_params"] = trainable_params
    stats["non_trainable_params"] = non_trainable_params

    if print_model:
        print_and_log(log, "\n" + "=" * 80)
        print_and_log(log, "MODEL STRUCTURE")
        print_and_log(log, "=" * 80)
        print_and_log(log, model)

    if print_params:
        print_and_log(log, "\n" + "=" * 80)
        print_and_log(log, "PARAMETER SUMMARY")
        print_and_log(log, "=" * 80)
        print_and_log(log, f"Total Parameters     : {total_params:,} ({format_num(total_params)})")
        print_and_log(log, f"Trainable Parameters : {trainable_params:,} ({format_num(trainable_params)})")
        print_and_log(log, f"Non-trainable Params : {non_trainable_params:,} ({format_num(non_trainable_params)})")

    if print_flops:
        if profile is not None and clever_format is not None:
            try:
                dummy_input = torch.randn(*input_shape).to(device)
                macs, params = profile(model, inputs=(dummy_input,), verbose=False)
                macs_str, params_str = clever_format([macs, params], "%.3f")
                stats["macs"] = float(macs)
                stats["flops"] = float(macs * 2)
                stats["thop_params"] = float(params)

                print_and_log(log, "\n" + "=" * 80)
                print_and_log(log, "FLOPs / MACs")
                print_and_log(log, "=" * 80)
                print_and_log(log, f"MACs           : {macs_str}")
                print_and_log(log, f"Params(thop)   : {params_str}")
                print_and_log(log, f"Approx. FLOPs  : ~ {macs * 2:.3e}")
            except Exception as e:
                print_and_log(log, f"FLOPs/MACs profiling failed: {e}")

    return stats


def save_loss_curve(train_losses, valid_losses, save_path, log=None):
    if len(train_losses) == 0 or len(valid_losses) == 0:
        return

    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    log_file_only(log, f'Loss curve saved to: {save_path}')


def safe_save_checkpoint(model_state_dict, save_path, log=None):
    try:
        ensure_parent_dir(save_path)
        torch.save(model_state_dict, save_path)
        log_file_only(log, f'Checkpoint saved successfully: {save_path}')
        return True
    except Exception as e:
        log_file_only(log, f'Checkpoint save failed: {e}')
        return False


def safe_load_checkpoint(model, ckpt_path, device, log=None):
    if not os.path.exists(ckpt_path):
        log_file_only(log, f'Best model file not found: {ckpt_path}')
        return False
    try:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        log_file_only(log, f'Checkpoint loaded successfully: {ckpt_path}')
        return True
    except Exception as e:
        log_file_only(log, f'Failed to load checkpoint: {e}')
        return False


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_inverse_transform(scaler, x):
    x_np = to_numpy(x)
    try:
        out = scaler.inverse_transform(x_np)
        return to_numpy(out), True, None
    except Exception as e:
        return x_np, False, str(e)


def compute_metrics(y_pred, y_true):
    """
    Only return MAE and RMSE.
    Compatible with numpy.ndarray and torch.Tensor.
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    elif not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(np.asarray(y_pred), dtype=torch.float32)
    else:
        y_pred = y_pred.float()

    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    elif not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(np.asarray(y_true), dtype=torch.float32)
    else:
        y_true = y_true.float()

    y_true_np = y_true.detach().cpu().numpy().reshape(-1)
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)

    mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
    rmse = float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))

    return mae, rmse


set_seed(args.seed)

use_cuda = torch.cuda.is_available() and (not args.no_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

task_suffix = extract_data_suffix(args.data)
args.save_result = get_result_root_dir(args.dataset, task_suffix)

args.log_file = resolve_log_file_path(args.log_file, args.dataset, args.expid)
ensure_dir(args.save)
ensure_dir(args.save_loss)
ensure_dir(args.save_result)

log = open(args.log_file, 'w', encoding='utf-8')


def main():
    print_and_log(log, "=" * 80)
    print_and_log(log, "HYPERPARAMETERS")
    print_and_log(log, "=" * 80)
    print_and_log(log, args)

    static_adj_np = get_adjacency_matrix(
        distance_df_filename=args.sensors_distance,
        normalize=True
    )
    adj = torch.FloatTensor(static_adj_np).to(device)

    dataloader = load_dataset(
        dataset_dir=args.data,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        test_batch_size=args.test_batch_size,
        fill_zeros=args.fill_zeros
    )

    scaler = dataloader['scaler']

    engine = trainer(
        scaler=scaler,
        adj=adj,
        history=args.history,
        num_of_vertices=args.num_of_vertices,
        horizon=args.horizon,
        strides=args.strides,
        in_dim=args.in_dim,
        hidden_dims=args.hidden_dims,
        first_layer_embedding_size=args.first_layer_embedding_size,
        out_layer_dim=args.out_layer_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        factor=args.factor,
        attention_dropout=args.attention_dropout,
        output_attention=args.output_attention,
        dropout=args.dropout,
        forward_expansion=args.forward_expansion,
        use_adaptive_graph=args.use_adaptive_graph,
        lrate=args.learning_rate,
        w_decay=args.weight_decay,
        l_decay_rate=args.lr_decay_rate,
        lr_decay=args.lr_decay,
        max_grad_norm=args.max_grad_norm,
        log=log,
        device=device,
        activation=args.activation,
        use_mask=args.use_mask,
        temporal_emb=args.temporal_emb,
        spatial_emb=args.spatial_emb,
        use_transformer=args.use_transformer,
        use_informer=args.use_informer,
        loss_function=args.loss_function,
        huber_delta=args.huber_delta,
        norm_type=args.norm_type,
        ffn_activation=args.ffn_activation,
        return_stability_stats=args.return_stability_stats,
        use_temporal_inception=args.use_temporal_inception,
    )

    base_model = engine.model.module if hasattr(engine.model, "module") else engine.model

    input_shape = (1, args.history, args.num_of_vertices, args.in_dim)
    results_csv_path = get_results_csv_path(args.save_result, args.dataset, task_suffix)
    epoch_memory_csv_path = get_epoch_memory_csv_path(args.save_result, args.dataset, args.expid, task_suffix)
    memory_plot_path = get_memory_plot_path(args.save_result, args.dataset, args.expid, task_suffix)
    per_horizon_csv = get_per_horizon_csv_path(args.save_result, args.dataset, args.expid, task_suffix)
    test_debug_npz = get_test_debug_npz_path(args.save_result, args.dataset, args.expid, task_suffix)

    epoch_memory_records = []

    if args.print_model or args.print_params or args.print_flops:
        model_stats = print_model_statistics(
            model=base_model,
            input_shape=input_shape,
            device=device,
            print_model=args.print_model,
            print_params=args.print_params,
            print_flops=args.print_flops,
            log=log
        )
    else:
        model_stats = get_model_statistics_dict(base_model, input_shape, device)

    his_loss = []
    train_time = []
    val_time = []

    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_train_mae = []
    epoch_valid_mae = []
    epoch_train_rmse = []
    epoch_valid_rmse = []

    wait = 0
    best_val_loss = float('inf')
    best_model_path = get_best_model_path(args.save, args.expid)
    loss_curve_path = get_loss_curve_path(args.save_loss, args.expid)
    history_npz_path = get_history_npz_path(args.save_loss, args.expid)

    total_start_time = time.time()
    reset_gpu_peak_memory()

    print_and_log(log, "\n" + "=" * 80)
    print_and_log(log, "TRAINING START")
    print_and_log(log, "=" * 80)

    for epoch in tqdm.tqdm(range(1, args.epochs + 1), leave=False):
        if wait >= args.patience:
            print_and_log(log, f"Early stopping triggered at epoch {epoch - 1}.")
            break

        train_loss = []
        train_mae = []
        train_rmse = []

        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for x, y in dataloader['train_loader'].get_iterator():
            x_train = torch.tensor(x, dtype=torch.float32, device=device)
            y_train = torch.tensor(y[:, :, :, 0], dtype=torch.float32, device=device)

            loss, tmae, trmse = engine.train_model(x_train, y_train)

            train_loss.append(loss)
            train_mae.append(tmae)
            train_rmse.append(trmse)

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mae = []
        valid_rmse = []

        s1 = time.time()
        for x, y in dataloader['val_loader'].get_iterator():
            x_val = torch.tensor(x, dtype=torch.float32, device=device)
            y_val = torch.tensor(y[:, :, :, 0], dtype=torch.float32, device=device)

            vloss, vmae, vrmse = engine.eval_model(x_val, y_val)

            valid_loss.append(vloss)
            valid_mae.append(vmae)
            valid_rmse.append(vrmse)
        s2 = time.time()
        val_time.append(s2 - s1)

        if args.lr_decay:
            try:
                engine.lr_scheduler.step()
            except Exception as e:
                log_file_only(log, f'LR scheduler step skipped: {e}')

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)

        epoch_train_losses.append(mtrain_loss)
        epoch_valid_losses.append(mvalid_loss)
        epoch_train_mae.append(mtrain_mae)
        epoch_valid_mae.append(mvalid_mae)
        epoch_train_rmse.append(mtrain_rmse)
        epoch_valid_rmse.append(mvalid_rmse)

        memory_stats = get_gpu_memory_stats()
        epoch_memory_record = {
            "epoch": epoch,
            "allocated_mb": round(memory_stats["allocated_mb"], 4),
            "reserved_mb": round(memory_stats["reserved_mb"], 4),
            "max_allocated_mb": round(memory_stats["max_allocated_mb"], 4),
            "max_reserved_mb": round(memory_stats["max_reserved_mb"], 4),
            "train_time_sec": round((t2 - t1), 4),
            "infer_time_sec": round((s2 - s1), 4),
            "train_loss": round(float(mtrain_loss), 6),
            "valid_loss": round(float(mvalid_loss), 6),
            "train_mae": round(float(mtrain_mae), 6),
            "valid_mae": round(float(mvalid_mae), 6),
            "train_rmse": round(float(mtrain_rmse), 6),
            "valid_rmse": round(float(mvalid_rmse), 6),
        }
        epoch_memory_records.append(epoch_memory_record)

        if epoch % args.print_every == 0 or epoch == 1:
            print_and_log(
                log,
                f"Epoch: {epoch:03d}/{args.epochs:03d} | "
                f"Train Loss: {mtrain_loss:.6f}, Train MAE: {mtrain_mae:.6f}, Train RMSE: {mtrain_rmse:.6f} | "
                f"Valid Loss: {mvalid_loss:.6f}, Valid MAE: {mvalid_mae:.6f}, Valid RMSE: {mvalid_rmse:.6f} | "
                f"Train Time: {t2 - t1:.4f}s, Valid Time: {s2 - s1:.4f}s"
            )

        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            wait = 0
            safe_save_checkpoint(engine.model.state_dict(), best_model_path, log=log)
            print_and_log(log, f"--> New best model saved! Best Val Loss: {best_val_loss:.6f}")
        else:
            wait += 1
            print_and_log(log, f"--> No improvement. EarlyStopping counter: {wait}/{args.patience}")

        np.save(os.path.join(args.save_loss, f'history_loss_{args.expid}.npy'), np.array(his_loss))

        np.savez(
            history_npz_path,
            train_loss=np.array(epoch_train_losses),
            valid_loss=np.array(epoch_valid_losses),
            train_mae=np.array(epoch_train_mae),
            valid_mae=np.array(epoch_valid_mae),
            train_rmse=np.array(epoch_train_rmse),
            valid_rmse=np.array(epoch_valid_rmse),
        )

        save_loss_curve(epoch_train_losses, epoch_valid_losses, loss_curve_path, log=log)

    if len(epoch_train_losses) > 0 and len(epoch_valid_losses) > 0:
        save_loss_curve(epoch_train_losses, epoch_valid_losses, loss_curve_path, log=log)

    save_epoch_memory_csv(epoch_memory_csv_path, epoch_memory_records)
    plot_memory_vs_epoch(epoch_memory_records, memory_plot_path, log=log)

    final_memory_stats = get_gpu_memory_stats()
    avg_train_time = np.mean(train_time) if len(train_time) > 0 else 0.0
    avg_infer_time = np.mean(val_time) if len(val_time) > 0 else 0.0
    trainable_params = model_stats.get("trainable_params", 0) or 0
    flops_value = model_stats.get("flops", None)
    peak_gpu_memory_mb = final_memory_stats.get("max_allocated_mb", 0.0)
    total_time = time.time() - total_start_time

    if not safe_load_checkpoint(engine.model, best_model_path, device=device, log=log):
        raise FileNotFoundError(f"Best checkpoint not found or failed to load: {best_model_path}")


    outputs = []
    realy_list = []

    for x, y in dataloader['test_loader'].get_iterator():
        x_test = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = engine.model(x_test)

        outputs.append(preds.detach().cpu())
        realy_list.append(torch.tensor(y[:, :, :, 0], dtype=torch.float32))

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(realy_list, dim=0)

    yhat_np = to_numpy(yhat)
    realy_np = to_numpy(realy)

    yhat_denorm, _, _ = safe_inverse_transform(scaler, yhat_np)
    realy_eval = realy_np

    np.savez(
        test_debug_npz,
        yhat=yhat_np,
        realy=realy_np,
        yhat_denorm=yhat_denorm
    )

    amae, armse = [], []
    per_horizon_rows = []

    print_and_log(log, "\n" + "=" * 80)
    print_and_log(log, "FINAL RESULT")
    print_and_log(log, "=" * 80)

    for i in range(args.horizon):
        pred_i = yhat_denorm[:, i, :]
        real_i = realy_eval[:, i, :]

        mae, rmse = compute_metrics(pred_i, real_i)

        amae.append(mae)
        armse.append(rmse)

        per_horizon_rows.append({
            "horizon": i + 1,
            "mae": round(float(mae), 6),
            "rmse": round(float(rmse), 6),
            "scale": "pred_denorm_real_raw"
        })

        print_and_log(
            log,
            f"Evaluate best model on test data for horizon {i + 1}, "
            f"Test MAE: {mae:.6f}, Test RMSE: {rmse:.6f}, "
            f"Scale: pred_denorm_real_raw"
        )

    mean_mae = float(np.mean(amae))
    mean_rmse = float(np.mean(armse))

    print_and_log(
        log,
        f"On average over {args.horizon} horizons, "
        f"Test MAE: {mean_mae:.6f}, Test RMSE: {mean_rmse:.6f}"
    )

    print_and_log(log, "\n" + "=" * 80)
    print_and_log(log, "EFFICIENCY")
    print_and_log(log, "=" * 80)
    print_and_log(log, f"Average Training Time: {float(avg_train_time):.4f} sec/epoch")
    print_and_log(log, f"Average Inference Time: {float(avg_infer_time):.4f} sec")
    print_and_log(log, f"Trainable Parameters: {int(trainable_params):,} ({float(trainable_params) / 1e3:.4f} K)")
    if flops_value is not None:
        print_and_log(log, f"FLOPs: {float(flops_value):.4f} ({format_flops_value(float(flops_value))})")
    else:
        print_and_log(log, "FLOPs: N/A")
    print_and_log(log, f"Peak GPU Memory Usage: {float(peak_gpu_memory_mb):.2f} MB")
    print_and_log(log, f"Total time: {float(total_time):.6f}")
    print_and_log(log, "=" * 80)

    save_per_horizon_csv(per_horizon_csv, per_horizon_rows)

    final_results = {
        "dataset": args.dataset,
        "task_suffix": task_suffix,
        "expid": args.expid,
        "seed": args.seed,
        "history": args.history,
        "horizon": args.horizon,
        "strides": args.strides,
        "batch_size": args.batch_size,
        "valid_batch_size": args.valid_batch_size,
        "test_batch_size": args.test_batch_size,
        "num_of_vertices": args.num_of_vertices,
        "in_dim": args.in_dim,
        "hidden_dims": str(args.hidden_dims),
        "first_layer_embedding_size": args.first_layer_embedding_size,
        "out_layer_dim": args.out_layer_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "forward_expansion": args.forward_expansion,
        "use_adaptive_graph": args.use_adaptive_graph,
        "temporal_emb": args.temporal_emb,
        "spatial_emb": args.spatial_emb,
        "use_transformer": args.use_transformer,
        "use_informer": args.use_informer,
        "factor": args.factor,
        "attention_dropout": args.attention_dropout,
        "output_attention": args.output_attention,
        "use_mask": args.use_mask,
        "activation": args.activation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_decay": args.lr_decay,
        "lr_decay_rate": args.lr_decay_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "max_grad_norm": args.max_grad_norm,
        "loss_function": args.loss_function,
        "huber_delta": args.huber_delta,
        "norm_type": args.norm_type,
        "ffn_activation": args.ffn_activation,
        "use_temporal_inception": args.use_temporal_inception,
        "best_val_loss": round(float(best_val_loss), 6),
        "avg_test_mae": round(float(mean_mae), 6),
        "avg_test_rmse": round(float(mean_rmse), 6),
        "avg_train_time_sec_per_epoch": round(float(avg_train_time), 6),
        "avg_infer_time_sec": round(float(avg_infer_time), 6),
        "trainable_params": int(trainable_params),
        "flops": None if flops_value is None else float(flops_value),
        "peak_gpu_memory_mb": round(float(peak_gpu_memory_mb), 4),
        "total_time_sec": round(float(total_time), 6),
        "log_file": args.log_file,
        "best_model_path": best_model_path,
        "loss_curve_path": loss_curve_path,
        "per_horizon_csv": per_horizon_csv,
        "test_debug_npz": test_debug_npz
    }


    log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        if log is not None and not log.closed:
            log_file_only(log, f"Fatal error: {e}")
            log.close()
        raise