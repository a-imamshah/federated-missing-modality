"""ClientApp: local training on this client's share of the data."""
from types import SimpleNamespace

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedmm.task import get_setup

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    cfg = context.run_config
    partition_id = context.node_config["partition-id"]
    setup = get_setup(cfg, context.node_config["num-partitions"])
    client = setup.clients[partition_id]

    from federated import train_client
    from model import Net

    torch.set_num_threads(1)
    torch.manual_seed(int(cfg["seed"]) * 1_000_003 + msg.content["config"]["server-round"] * 1009 + partition_id)
    model = Net(dim=setup.dim, num_classes=setup.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    args = SimpleNamespace(lr=cfg["learning-rate"], local_epochs=cfg["local-epochs"],
                           batch_size=cfg["batch-size"], modality_dropout=cfg["modality-dropout"])
    state = train_client(model, client, args)

    content = RecordDict({"arrays": ArrayRecord(state), "metrics": MetricRecord({"num-examples": len(client.y)})})
    return Message(content=content, reply_to=msg)
