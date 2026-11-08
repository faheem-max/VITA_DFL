import argparse
import csv
import json
import os
import shutil

import numpy as np
import torch
from torchvision import datasets


CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
NUM_CLASSES = 10


def normalize_images(images):
    # CIFAR-10 is stored as N x H x W x C (uint8).
    # Convert to N x C x H x W, scale to [0, 1], then normalize.
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2).float().div(255.0)
    return (images - CIFAR10_MEAN) / CIFAR10_STD


def label_distribution(labels):
    counts = torch.bincount(labels.long(), minlength=NUM_CLASSES)
    return {str(i): int(counts[i]) for i in range(NUM_CLASSES)}


def save_tensor_dataset(path, images, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "images": images.cpu(),
            "labels": labels.long().cpu(),
        },
        path,
    )


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def balanced_global_validation_split(labels, global_val_size, generator):
    if global_val_size <= 0 or global_val_size >= len(labels):
        raise ValueError("Invalid global_val_size.")

    base = global_val_size // NUM_CLASSES
    remainder = global_val_size % NUM_CLASSES

    val_parts = []
    pool_parts = []

    for class_id in range(NUM_CLASSES):
        idx = torch.where(labels == class_id)[0]
        idx = idx[torch.randperm(len(idx), generator=generator)]

        n_val = base + (1 if class_id < remainder else 0)

        val_parts.append(idx[:n_val])
        pool_parts.append(idx[n_val:])

    val_idx = torch.cat(val_parts)
    pool_idx = torch.cat(pool_parts)

    val_idx = val_idx[
        torch.randperm(len(val_idx), generator=generator)
    ]

    return val_idx, pool_idx


def largest_remainder_capacities(total_samples, client_ids, throughputs):
    weights = np.asarray(
        [throughputs[cid] for cid in client_ids],
        dtype=np.float64,
    )

    if np.any(weights <= 0):
        raise ValueError("All throughputs must be > 0.")

    raw = weights / weights.sum() * total_samples
    counts = np.floor(raw).astype(np.int64)

    remaining = int(total_samples - counts.sum())
    fractions = raw - counts
    order = np.argsort(-fractions)

    for idx in order[:remaining]:
        counts[idx] += 1

    capacities = {
        client_ids[i]: int(counts[i])
        for i in range(len(client_ids))
    }

    raw_capacities = {
        client_ids[i]: float(raw[i])
        for i in range(len(client_ids))
    }

    if sum(capacities.values()) != total_samples:
        raise RuntimeError("Capacity rounding failed.")

    return capacities, raw_capacities


def throughput_weighted_dirichlet_partition(
    pool_indices,
    labels,
    client_ids,
    capacities,
    alpha,
    seed,
):
    """
    Capacity-constrained class-wise Dirichlet partition.

    For each class k:
        p_k ~ Dirichlet(alpha, ..., alpha)

    Total samples per client are fixed by measured throughput capacity.
    Label composition remains non-IID.
    """
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be > 0.")

    num_clients = len(client_ids)

    if sum(capacities.values()) != len(pool_indices):
        raise ValueError(
            "Client capacities must sum exactly to the client pool size."
        )

    rng = np.random.default_rng(seed + 2026)

    class_client_probs = rng.dirichlet(
        alpha=np.full(num_clients, alpha, dtype=np.float64),
        size=NUM_CLASSES,
    )

    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed + 3030)

    pool_labels = labels[pool_indices]

    class_indices = {}
    class_ptr = {}

    for class_id in range(NUM_CLASSES):
        idx = pool_indices[torch.where(pool_labels == class_id)[0]]
        idx = idx[torch.randperm(len(idx), generator=torch_gen)]
        class_indices[class_id] = idx.tolist()
        class_ptr[class_id] = 0

    class_sequence = []
    for class_id in range(NUM_CLASSES):
        class_sequence.extend([class_id] * len(class_indices[class_id]))

    class_sequence = np.asarray(class_sequence, dtype=np.int64)
    rng.shuffle(class_sequence)

    remaining = np.asarray(
        [capacities[cid] for cid in client_ids],
        dtype=np.int64,
    )

    assignments = {cid: [] for cid in client_ids}
    realized = {
        cid: np.zeros(NUM_CLASSES, dtype=np.int64)
        for cid in client_ids
    }

    for class_id in class_sequence:
        class_id = int(class_id)

        weights = (
            class_client_probs[class_id]
            * remaining.astype(np.float64)
        )

        total_weight = float(weights.sum())

        if total_weight <= 0:
            weights = remaining.astype(np.float64)
            total_weight = float(weights.sum())

        if total_weight <= 0:
            raise RuntimeError(
                "No client capacity remains while samples are unassigned."
            )

        probs = weights / total_weight
        pos = int(rng.choice(num_clients, p=probs))
        client_id = client_ids[pos]

        ptr = class_ptr[class_id]
        sample_index = class_indices[class_id][ptr]
        class_ptr[class_id] += 1

        assignments[client_id].append(int(sample_index))
        realized[client_id][class_id] += 1
        remaining[pos] -= 1

    if int(remaining.sum()) != 0:
        raise RuntimeError("Partition did not fill all client capacities.")

    for cid in client_ids:
        if len(assignments[cid]) != capacities[cid]:
            raise RuntimeError(
                f"Client {cid}: expected {capacities[cid]}, "
                f"got {len(assignments[cid])}"
            )

    return assignments, class_client_probs, realized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_hardware.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config.get("seed", 42))
    num_clients = int(config.get("num_clients", 8))
    num_cliques = int(config.get("num_cliques", 2))
    clique_size = int(config.get("clique_size", 4))

    if num_clients != 8:
        raise ValueError(
            "For this physical experiment set num_clients=8. "
            "Physical IDs are still 1..8."
        )

    if num_cliques != 2 or clique_size != 4:
        raise ValueError(
            "For this experiment use num_cliques=2 and clique_size=4."
        )

    client_ids = list(range(1, num_clients + 1))

    alpha = float(config.get("dirichlet_alpha", 0.30))
    global_val_size = int(config.get("global_val_size", 5000))
    local_val_fraction = float(
        config.get("local_val_fraction", 0.10)
    )

    if not 0 < local_val_fraction < 1:
        raise ValueError(
            "local_val_fraction must be between 0 and 1."
        )

    throughput_path = config.get(
        "throughput_profile_path",
        "throughput_profile.json",
    )

    device_map_path = config.get(
        "device_map_path",
        "device_map.json",
    )

    data_dir = config.get(
        "hardware_data_dir",
        "./dataset_hardware_cifar10",
    )

    with open(throughput_path, "r", encoding="utf-8") as f:
        throughput_profile = json.load(f)

    with open(device_map_path, "r", encoding="utf-8") as f:
        device_map = json.load(f)

    throughputs = {}

    for cid in client_ids:
        key = str(cid)

        if key not in throughput_profile["clients"]:
            raise KeyError(
                f"Client {cid} missing from throughput_profile.json"
            )

        value = float(
            throughput_profile["clients"][key][
                "throughput_samples_per_sec"
            ]
        )

        if value <= 0:
            raise ValueError(
                f"Client {cid} has invalid throughput {value}."
            )

        throughputs[cid] = value

    manifest_path = os.path.join(
        data_dir,
        "split_manifest.json",
    )

    if os.path.exists(manifest_path) and not args.force:
        raise RuntimeError(
            f"{data_dir} already contains a prepared split.\n"
            "Back it up first, or rerun with --force."
        )

    if args.force and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    raw_dir = os.path.join(data_dir, "raw_cifar10")
    clients_dir = os.path.join(data_dir, "clients")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clients_dir, exist_ok=True)

    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    cifar_train = datasets.CIFAR10(
        root=raw_dir,
        train=True,
        download=True,
    )

    cifar_test = datasets.CIFAR10(
        root=raw_dir,
        train=False,
        download=True,
    )

    train_images = normalize_images(cifar_train.data)
    train_labels = torch.tensor(cifar_train.targets, dtype=torch.long)

    test_images = normalize_images(cifar_test.data)
    test_labels = torch.tensor(cifar_test.targets, dtype=torch.long)

    global_val_indices, pool_indices = (
        balanced_global_validation_split(
            labels=train_labels,
            global_val_size=global_val_size,
            generator=generator,
        )
    )

    save_tensor_dataset(
        os.path.join(data_dir, "global_validation.pt"),
        train_images[global_val_indices].clone(),
        train_labels[global_val_indices].clone(),
    )

    save_tensor_dataset(
        os.path.join(data_dir, "global_test.pt"),
        test_images,
        test_labels,
    )

    capacities, raw_capacities = largest_remainder_capacities(
        total_samples=len(pool_indices),
        client_ids=client_ids,
        throughputs=throughputs,
    )

    assignments, sampled_probs, realized = (
        throughput_weighted_dirichlet_partition(
            pool_indices=pool_indices,
            labels=train_labels,
            client_ids=client_ids,
            capacities=capacities,
            alpha=alpha,
            seed=seed,
        )
    )

    rows = []
    client_manifest = {}

    for cid in client_ids:
        client_indices = torch.tensor(
            assignments[cid],
            dtype=torch.long,
        )

        client_gen = torch.Generator()
        client_gen.manual_seed(seed + 1000 + cid)

        perm = torch.randperm(
            len(client_indices),
            generator=client_gen,
        )

        client_indices = client_indices[perm]

        n_val = int(
            round(len(client_indices) * local_val_fraction)
        )

        n_val = max(
            1,
            min(n_val, len(client_indices) - 1),
        )

        local_val_idx = client_indices[:n_val]
        local_train_idx = client_indices[n_val:]

        local_train_images = train_images[local_train_idx].clone()
        local_train_labels = train_labels[local_train_idx].clone()

        local_val_images = train_images[local_val_idx].clone()
        local_val_labels = train_labels[local_val_idx].clone()

        client_dir = os.path.join(
            clients_dir,
            f"client_{cid}",
        )

        save_tensor_dataset(
            os.path.join(client_dir, "local_train.pt"),
            local_train_images,
            local_train_labels,
        )

        save_tensor_dataset(
            os.path.join(client_dir, "local_validation.pt"),
            local_val_images,
            local_val_labels,
        )

        assigned_labels = train_labels[client_indices]
        assigned_dist = label_distribution(assigned_labels)
        train_dist = label_distribution(local_train_labels)
        val_dist = label_distribution(local_val_labels)

        mapped = device_map["clients"][str(cid)]
        clique_id = int(mapped["clique_id"])

        client_manifest[str(cid)] = {
            "physical_client_id": cid,
            "device_name": mapped["device_name"],
            "hardware_type": mapped["hardware_type"],
            "clique_id": clique_id,
            "throughput_samples_per_sec": throughputs[cid],
            "raw_proportional_capacity": raw_capacities[cid],
            "num_total_assigned": len(client_indices),
            "num_local_train": len(local_train_idx),
            "num_local_validation": len(local_val_idx),
            "assigned_label_distribution": assigned_dist,
            "train_label_distribution": train_dist,
            "validation_label_distribution": val_dist,
        }

        row = {
            "client_id": cid,
            "device_name": mapped["device_name"],
            "hardware_type": mapped["hardware_type"],
            "clique_id": clique_id,
            "throughput_samples_per_sec": throughputs[cid],
            "total_assigned": len(client_indices),
            "local_train": len(local_train_idx),
            "local_validation": len(local_val_idx),
        }

        for class_id in range(NUM_CLASSES):
            row[f"label_{class_id}"] = assigned_dist[str(class_id)]

        rows.append(row)

        print(
            f"Client {cid} | "
            f"{mapped['hardware_type']:<20} | "
            f"throughput={throughputs[cid]:8.2f}/s | "
            f"total={len(client_indices):5d} | "
            f"train={len(local_train_idx):5d} | "
            f"val={len(local_val_idx):4d}"
        )

    save_csv(
        os.path.join(data_dir, "client_data_distribution.csv"),
        [
            "client_id",
            "device_name",
            "hardware_type",
            "clique_id",
            "throughput_samples_per_sec",
            "total_assigned",
            "local_train",
            "local_validation",
        ] + [f"label_{i}" for i in range(NUM_CLASSES)],
        rows,
    )

    clique_rows = []

    for clique_id in sorted(
        {
            int(device_map["clients"][str(cid)]["clique_id"])
            for cid in client_ids
        }
    ):
        members = [
            cid
            for cid in client_ids
            if int(
                device_map["clients"][str(cid)]["clique_id"]
            ) == clique_id
        ]

        row = {
            "clique_id": clique_id,
            "num_clients": len(members),
            "total_assigned": sum(
                client_manifest[str(cid)]["num_total_assigned"]
                for cid in members
            ),
        }

        for class_id in range(NUM_CLASSES):
            row[f"label_{class_id}"] = sum(
                client_manifest[str(cid)][
                    "assigned_label_distribution"
                ][str(class_id)]
                for cid in members
            )

        clique_rows.append(row)

    save_csv(
        os.path.join(data_dir, "clique_data_distribution.csv"),
        [
            "clique_id",
            "num_clients",
            "total_assigned",
        ] + [f"label_{i}" for i in range(NUM_CLASSES)],
        clique_rows,
    )

    manifest = {
        "seed": seed,
        "dataset": "CIFAR-10",
        "physical_client_id_scheme": "1..8",
        "num_clients": num_clients,
        "num_cliques": num_cliques,
        "clique_size": clique_size,
        "partition_type": (
            f"throughput-weighted capacity-constrained Dirichlet "
            f"alpha={alpha:g} label-skew non-IID"
        ),
        "dirichlet_alpha": alpha,
        "allocation_rule": (
            "Total client sample capacity proportional to measured "
            "median training throughput."
        ),
        "global_validation_size": len(global_val_indices),
        "global_test_size": len(test_labels),
        "client_pool_size": len(pool_indices),
        "local_validation_fraction": local_val_fraction,
        "global_validation_label_distribution": label_distribution(
            train_labels[global_val_indices]
        ),
        "global_test_label_distribution": label_distribution(test_labels),
        "clients": client_manifest,
        "sampled_dirichlet_class_to_client_proportions": {
            str(class_id): {
                str(cid): float(
                    sampled_probs[class_id, pos]
                )
                for pos, cid in enumerate(client_ids)
            }
            for class_id in range(NUM_CLASSES)
        },
        "realized_client_class_counts": {
            str(cid): {
                str(class_id): int(realized[cid][class_id])
                for class_id in range(NUM_CLASSES)
            }
            for cid in client_ids
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 76)
    print("CIFAR-10 physical-hardware non-IID split complete")
    print("=" * 76)
    print(f"Output directory : {data_dir}")
    print(f"Global validation: {len(global_val_indices)}")
    print(f"Client pool      : {len(pool_indices)}")
    print(f"Official test    : {len(test_labels)}")
    print(f"Dirichlet alpha  : {alpha}")
    print(
        "Assigned total   :",
        sum(capacities.values()),
    )
    print()
    print(
        "Next: inspect client_data_distribution.csv before copying "
        "any .pt files to devices."
    )


if __name__ == "__main__":
    main()
