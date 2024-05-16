import torch.optim.lr_scheduler
from src.exp_default_settings import *
import torch.optim as optim
from src.baselines import *
from src.datasets import *
from src.networks import (
    WideResNet,
    DenseNet121_CE,
    LinearNet,
    LinearNetDefer,
    NonLinearNet,
    ResNet50_CE,
)
import os
import argparse
import datetime
from tqdm import tqdm
from codecarbon import track_emissions

torch.autograd.set_detect_anomaly(True)


def get_raw_res(data):
    tmp = pd.DataFrame()
    tmp["rej_score"] = data["rej_score"]
    tmp["labels"] = data["labels"]
    tmp["hum_preds"] = data["hum_preds"]
    tmp["preds"] = data["preds"]
    for i in range(data["class_probs"][0].shape[0]):
        tmp["class_probs_{}".format(i)] = data["class_probs"][:, i]
    return tmp


def load_model_trained(n, path, device, dataset, class_network="Linear"):
    if "chestxray" in dataset:
        model_cnn = DenseNet121_CE(2).to(device)
        # torch load
        model_cnn.load_state_dict(torch.load(path, map_location=device))
        for param in model_cnn.parameters():
            param.requires_grad = False
        if class_network == "Linear":
            model_cnn.densenet121.classifier = nn.Linear(model_cnn.num_ftrs, n).to(
                device
            )
        elif class_network == "NonLinear":
            model_cnn.densenet121.classifier = NonLinearNet(model_cnn.num_ftrs, n).to(
                device
            )
    elif dataset == "cifar10h":
        model_cnn = WideResNet(28, 10, 4, dropRate=0).to(device)
        # torch load
        model_cnn.load_state_dict(torch.load(path, map_location=device))
        for param in model_cnn.parameters():
            param.requires_grad = False
        model_cnn.fc2 = nn.Linear(50, n).to(device)
        # torch load

    # require grad
    return model_cnn


def load_model_imagenet(n, device):
    model_linear = DenseNet121_CE(n).to(device)
    for param in model_linear.parameters():
        param.requires_grad = False
    model_linear.densenet121.classifier.requires_grad_(True)
    return model_linear


def load_model_galaxyzoo(n, device):
    model_cnn = ResNet50_CE(n).to(device)
    return model_cnn


@track_emissions(
    project_name="RDD",
    measure_power_secs=90,
    api_call_interval=5,
    output_file="RDD_emissions.csv",
)
def test(
    seed=seed,
    defer_system=defer_system,
    data=data,
    data_distribution=data_distribution,
    expert_deferred_error=expert_deferred_error,
    expert_nondeferred_error=expert_nondeferred_error,
    machine_nondeferred_error=machine_nondeferred_error,
    d=d,
    num_of_guassians=num_of_guassians,
    device=device,
    label_chosen=label_chosen,
    ts=10000,
    target_coverages=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, None],
):
    if not os.path.exists("../exp_data"):
        os.makedirs("../exp_data")
        os.makedirs("../exp_data/data")
        os.makedirs("../exp_data/plots")
    else:
        if not os.path.exists("../exp_data/data"):
            os.makedirs("../exp_data/data")
        if not os.path.exists("../exp_data/plots"):
            os.makedirs("../exp_data/plots")
    date_now = datetime.datetime.now()
    # date_now = date_now.strftime("%Y-%m-%d_%H%M%S")
    set_seed(seed)
    # ns = [10000]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    m_norm = 5
    # generate data
    if os.path.exists("models/") is False:
        os.mkdir("models/")
    if os.path.exists("models/{}/".format(data)) is False:
        os.mkdir("models/{}/".format(data))
    if data == "synth":
        train_size = ts
        # optimizer = optim.Adam
        # scheduler = torch.optim.lr_scheduler.StepLR
        s_size = 25
        g = 0.5
        optimizer = optim.Adam
        scheduler = None
        lr = 0.01
        total_epochs = 50
        n_class = 2
        w_classes = None
        m_n = 5
        dataset = SyntheticData(
            train_samples=20000,
            test_samples=5000,
            data_distribution=data_distribution,
            d=d,
            mean_scale=1,
            expert_deferred_error=expert_deferred_error,
            expert_nondeferred_error=expert_nondeferred_error,
            machine_nondeferred_error=machine_nondeferred_error,
            num_of_guassians=num_of_guassians,
            val_split=0.1,
            batch_size=1024,
        )
        # instatiate model base
        if defer_system in ["RS", "MoE", "LCE", "OVA", "ASM"]:
            # model_base = LinearNet(dataset.d, num_class).to(device)
            model_base = LinearNetDefer(dataset.d, n_class).to(device)
        elif defer_system in ["CC"]:
            model_class = LinearNet(dataset.d, 2).to(device)
            model_expert = LinearNet(dataset.d, 2).to(device)
        elif defer_system == "SP":
            model_base = LinearNet(dataset.d, 2).to(device)
        elif defer_system == "DT":
            model_class = LinearNet(dataset.d, 2).to(device)
            model_rejector = LinearNet(dataset.d, 2).to(device)
        path_model = "models/{}/model_{}_{}_{}_{}_{}_{}_ep{}.pt".format(
            data,
            data,
            defer_system,
            seed,
            expert_deferred_error,
            expert_nondeferred_error,
            machine_nondeferred_error,
            total_epochs,
        )
        path_model_rej = "models/{}/model_rej_{}_{}_{}_{}_{}_{}_ep{}.pt".format(
            data,
            data,
            defer_system,
            seed,
            expert_deferred_error,
            expert_nondeferred_error,
            machine_nondeferred_error,
            total_epochs,
        )
        path_model_class = "models/{}/model_class_{}_{}_{}_{}_{}_{}_ep{}.pt".format(
            data,
            data,
            defer_system,
            seed,
            expert_deferred_error,
            expert_nondeferred_error,
            machine_nondeferred_error,
            total_epochs,
        )
        path_model_expert = "models/{}/model_expert_{}_{}_{}_{}_{}_{}_ep{}.pt".format(
            data,
            data,
            defer_system,
            seed,
            expert_deferred_error,
            expert_nondeferred_error,
            machine_nondeferred_error,
            total_epochs,
        )
    elif "chestxray" in data:
        if os.path.exists("models/chestxray/") is False:
            os.mkdir("models/chestxray")
        epochs = 10
        bs = 32
        path_model_pre = (
            "models/chestxray/chextxray_dn121_GC_{}_w1_lb{}_ep{}_bs{}_.pt".format(
                seed, label_chosen, epochs, bs
            )
        )
        # check if file exists and if not, train model
        if not os.path.isfile(path_model_pre):
            logging.info("training model on NIH training set for pretraining")
            model_cnn = DenseNet121_CE(2).to(device)
            optimizer_linear = optim.AdamW
            lr = 0.0001
            print(label_chosen)
            synth_data = ChestXrayDataset(
                True, True, data_dir="data", label_chosen=label_chosen, batch_size=bs
            )
            num_samples = len(synth_data.data_train_loader.dataset.targets)
            num_pos = sum(synth_data.data_train_loader.dataset.targets)
            num_neg = num_samples - num_pos
            w_classes = torch.Tensor([num_samples / num_neg, num_samples / num_pos]).to(
                device
            )
            # weights added to avoid forecasting only a single label in the pre-trained model
            print(w_classes)
            SP = SelectivePrediction(model_cnn, device, plotting_interval=1)
            SP.fit(
                synth_data.data_train_loader,
                synth_data.data_val_loader,
                synth_data.data_test_loader,
                epochs=epochs,
                optimizer=optimizer_linear,
                lr=lr,
                verbose=True,
                test_interval=1,
                weight=w_classes,
                m_norm=5,
            )
            sp_metrics = compute_classification_metrics(
                SP.test(synth_data.data_test_loader)
            )
            print(sp_metrics)
            # pickle the state_dict
            torch.save(SP.model_class.state_dict(), path_model_pre)
        set_seed(seed)
        optimizer = optim.AdamW
        scheduler = None
        lr = 1e-3
        total_epochs = 3  # 100
        dataset = ChestXrayDataset(
            False,
            True,
            data_dir="data",
            label_chosen=label_chosen,
            batch_size=128,
            test_split=0.2,
            val_split=0.10,
        )
        n_class = 2
        s_size = 25
        g = 0.5
        num_samples = len(dataset.data_train_loader.dataset.targets)
        num_pos = sum(dataset.data_train_loader.dataset.targets)
        num_neg = num_samples - num_pos
        # w_classes = torch.Tensor([num_samples / num_neg, num_samples / num_pos]).to(device)
        w_classes = None
        print(w_classes)
        # instatiate model base
        if defer_system in ["RS", "MoE", "LCE", "OVA", "ASM"]:
            # model_base = LinearNet(dataset.d, num_class).to(device)
            model_base = load_model_trained(
                n_class + 1, path_model_pre, device, data, class_network="NonLinear"
            )
        elif defer_system in ["CC"]:
            model_class = load_model_trained(
                n_class, path_model_pre, device, data, class_network="NonLinear"
            )
            model_expert = load_model_trained(
                2, path_model_pre, device, data, class_network="NonLinear"
            )
        elif defer_system == "SP":
            model_base = load_model_trained(
                n_class, path_model_pre, device, data, class_network="NonLinear"
            )
        elif defer_system == "DT":
            model_class = load_model_trained(
                n_class, path_model_pre, device, data, class_network="NonLinear"
            )
            model_rejector = load_model_trained(
                2, path_model_pre, device, data, class_network="NonLinear"
            )
        path_model = "models/{}/densenet121NL_model_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_rej = "models/{}/densenet121NL_model_rej_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_class = (
            "models/{}/densenet121NL_model_class_{}_{}_{}_ep{}.pt".format(
                data, data, defer_system, seed, total_epochs
            )
        )
        path_model_expert = (
            "models/{}/densenet121NL_model_expert_{}_{}_{}_ep{}.pt".format(
                data, data, defer_system, seed, total_epochs
            )
        )
    elif data == "cifar10h":
        path_model_pre = "models/cifar10h/cifar10h_wrn28_4_GC_200epochs.pt".format(seed)
        # check if file exists and if not, train model
        n_class = 10
        w_classes = None
        if not os.path.isfile(path_model_pre):
            logging.info("training model on cifar-10 for pretraining")
            model_cnn = WideResNet(28, 10, 4, dropRate=0).to(device)
            epochs = 3
            optimizer_linear = optim.AdamW
            lr = 0.001
            synth_data = CifarSynthDataset(3, True, batch_size=128, val_split=0.05)
            SP = SelectivePrediction(model_cnn, device, plotting_interval=2)
            sp_metrics = SP.fit(
                synth_data.data_train_loader,
                synth_data.data_val_loader,
                synth_data.data_test_loader,
                epochs=200,
                optimizer=optimizer_linear,
                lr=lr,
                verbose=True,
                test_interval=2,
            )
            print(sp_metrics)
            # pickle the state_dict
            torch.save(SP.model_class.state_dict(), path_model_pre)
        set_seed(seed)
        optimizer = optim.AdamW
        scheduler = None
        s_size = 25
        g = 0.5
        lr = 1e-3
        max_trials = 10
        total_epochs = 150
        dataset = Cifar10h(
            False, data_dir="data/", batch_size=128, test_split=0.2, val_split=0.10
        )
        # instatiate model base
        if defer_system in ["RS", "MoE", "LCE", "OVA", "ASM"]:
            # model_base = LinearNet(dataset.d, num_class).to(device)
            model_base = load_model_trained(
                n_class + 1, path_model_pre, device, dataset=data
            )
        elif defer_system in ["CC"]:
            model_class = load_model_trained(
                n_class, path_model_pre, device, dataset=data
            )
            model_expert = load_model_trained(2, path_model_pre, device, dataset=data)
        elif defer_system == "SP":
            model_base = load_model_trained(
                n_class, path_model_pre, device, dataset=data
            )
        elif defer_system == "DT":
            model_class = load_model_trained(
                n_class, path_model_pre, device, dataset=data
            )
            model_rejector = load_model_trained(2, path_model_pre, device, dataset=data)
        path_model = "models/{}/wresenet_model_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_rej = "models/{}/wresenet_model_rej_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_class = "models/{}/wresenet_model_class_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_expert = "models/{}/wresenet_model_expert_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
    elif data == "hatespeech":
        optimizer = optim.Adam
        scheduler = None
        lr = 1e-2
        s_size = 25
        g = 0.5
        dataset = HateSpeech(
            "data/",
            True,
            False,
            "random_annotator",
            device,
            test_split=0.2,
            val_split=0.10,
            batch_size=128,
        )
        n_class = 3
        total_epochs = 100
        w_classes = None
        # instatiate model base
        if defer_system in ["RS", "MoE", "LCE", "OVA", "ASM"]:
            # model_base = LinearNet(dataset.d, num_class).to(device)
            model_base = LinearNetDefer(dataset.d, n_class).to(device)
        elif defer_system in ["CC"]:
            model_class = LinearNet(dataset.d, n_class).to(device)
            model_expert = LinearNet(dataset.d, 2).to(device)
        elif defer_system == "SP":
            model_base = LinearNet(dataset.d, n_class).to(device)
        elif defer_system == "DT":
            model_class = LinearNet(dataset.d, n_class).to(device)
            model_rejector = LinearNet(dataset.d, 2).to(device)
        path_model = "models/{}/linear_model_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_rej = "models/{}/linear_model_rej_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_class = "models/{}/linear_model_class_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_expert = "models/{}/linear_model_expert_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
    elif data == "galaxyzoo":
        optimizer = optim.Adam
        scheduler = None
        lr = 1e-3
        s_size = 25
        g = 0.5
        dataset = GalaxyZoo(
            data_dir="data/", batch_size=128, test_split=0.2, val_split=0.10
        )
        n_class = 2
        total_epochs = 50
        w_classes = None
        # instatiate model base
        if defer_system in ["RS", "MoE", "LCE", "OVA", "ASM"]:
            # model_base = LinearNet(dataset.d, num_class).to(device)
            model_base = load_model_galaxyzoo(n_class + 1, device)
        elif defer_system in ["CC"]:
            model_class = load_model_galaxyzoo(n_class, device)
            model_expert = load_model_galaxyzoo(2, device)
        elif defer_system == "SP":
            model_base = load_model_galaxyzoo(n_class, device)
        elif defer_system == "DT":
            model_class = load_model_galaxyzoo(n_class, device)
            model_rejector = load_model_galaxyzoo(2, device)
        path_model = "models/{}/resnet50_model_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_rej = "models/{}/resnet50_model_rej_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_class = "models/{}/resnet50_model_class_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
        path_model_expert = "models/{}/resnet50_model_expert_{}_{}_{}_ep{}.pt".format(
            data, data, defer_system, seed, total_epochs
        )
    print_time = False
    if defer_system == "RS":
        model = RealizableSurrogate(1, 5, model_base, device, True)
        if os.path.exists(path_model):
            model.model.to(device)
            model.model.load_state_dict(torch.load(path_model, map_location=device))
        else:
            if data != "synth":
                path_model_rs_hp = "models/{}/net_rs_hp_{}_{}_{}_ep{}".format(
                    data, data, defer_system, seed, total_epochs
                )
            else:
                path_model_rs_hp = "models/{}/net_rs_hp_{}_{}_{}_{}_{}_{}_ep{}".format(
                    data,
                    data,
                    defer_system,
                    seed,
                    expert_deferred_error,
                    expert_nondeferred_error,
                    machine_nondeferred_error,
                    total_epochs,
                )
            print_time = True
            start_time = time.time()
            model.fit_hyperparam(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                path_model_save=path_model_rs_hp,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model.state_dict(), path_model)
    elif defer_system == "MoE":
        model = MixtureOfExperts(model_base, device)
        if os.path.exists(path_model):
            model.model.to(device)
            model.model.load_state_dict(torch.load(path_model, map_location=device))
        else:
            print_time = True
            start_time = time.time()
            model.fit(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model.state_dict(), path_model)
    elif defer_system == "LCE":
        model = LceSurrogate(1, 5, model_base, device)
        if os.path.exists(path_model):
            model.model.to(device)
            model.model.load_state_dict(torch.load(path_model, map_location=device))
        else:
            if data != "synth":
                path_model_lce_hp = "models/{}/net_lce_hp_{}_{}_{}_ep{}".format(
                    data, data, defer_system, seed, total_epochs
                )
            else:
                path_model_lce_hp = (
                    "models/{}/net_lce_hp_{}_{}_{}_{}_{}_{}_ep{}".format(
                        data,
                        data,
                        defer_system,
                        seed,
                        expert_deferred_error,
                        expert_nondeferred_error,
                        machine_nondeferred_error,
                        total_epochs,
                    )
                )
            print_time = True
            start_time = time.time()
            model.fit_hyperparam(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                path_model_save=path_model_lce_hp,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model.state_dict(), path_model)
    elif defer_system == "CC":
        model = CompareConfidence(model_class, model_expert, device)
        if os.path.exists(path_model_class):
            model.model_class.to(device)
            model.model_expert.to(device)
            model.model_class.load_state_dict(
                torch.load(path_model_class, map_location=device)
            )
            model.model_expert.load_state_dict(
                torch.load(path_model_expert, map_location=device)
            )
        else:
            print_time = True
            start_time = time.time()
            model.fit(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model_class.state_dict(), path_model_class)
            torch.save(model.model_expert.state_dict(), path_model_expert)
    elif defer_system == "OVA":
        model = OVASurrogate(1, 5, model_base, device)
        if os.path.exists(path_model):
            model.model.to(device)
            model.model.load_state_dict(torch.load(path_model, map_location=device))
        else:
            print_time = True
            start_time = time.time()
            model.fit(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model.state_dict(), path_model)
    elif defer_system == "ASM":
        model = AsymmetricLCESurrogate(1, 5, model_base, device)
        if os.path.exists(path_model):
            model.model.to(device)
            model.model.load_state_dict(torch.load(path_model, map_location=device))
        else:
            print_time = True
            start_time = time.time()
            model.fit(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model.state_dict(), path_model)
    elif defer_system == "SP":
        model = SelectivePrediction(model_base, device)
        if os.path.exists(path_model_class):
            model.model_class.to(device)
            model.model_class.load_state_dict(
                torch.load(path_model_class, map_location=device)
            )
        else:
            print_time = True
            start_time = time.time()
            model.fit(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model_class.state_dict(), path_model_class)
    elif defer_system == "DT":
        model = DifferentiableTriage(
            model_class, model_rejector, device, 0.000, "human_error"
        )
        if os.path.exists(path_model_class):
            model.model_class.to(device)
            model.model_class.load_state_dict(
                torch.load(path_model_class, map_location=device)
            )
            model.model_rejector.to(device)
            model.model_rejector.load_state_dict(
                torch.load(path_model_rej, map_location=device)
            )
        else:
            if data != "synth":
                path_model_dt_hp = "models/{}/net_dt_hp_{}_{}_{}_ep{}".format(
                    data, data, defer_system, seed, total_epochs
                )
            else:
                path_model_dt_hp = "models/{}/net_dt_hp_{}_{}_{}_{}_{}_{}_ep{}".format(
                    data,
                    data,
                    defer_system,
                    seed,
                    expert_deferred_error,
                    expert_nondeferred_error,
                    machine_nondeferred_error,
                    total_epochs,
                )
            print_time = True
            start_time = time.time()
            model.fit_hyperparam(
                dataset.data_train_loader,
                dataset.data_val_loader,
                dataset.data_test_loader,
                epochs=total_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=lr,
                verbose=False,
                test_interval=2,
                step_size=s_size,
                gamma=g,
                path_model_save=path_model_dt_hp,
                weight=w_classes,
                m_norm=m_norm,
            )
            end_time = time.time()
            time_to_fit = end_time - start_time
            torch.save(model.model_class.state_dict(), path_model_class)
            torch.save(model.model_rejector.state_dict(), path_model_rej)
    test_vals = model.test(dataset.data_test_loader)
    calib_vals = model.test(dataset.data_val_loader)
    # here we apply np.exp to avoid having negative cutoffs
    res = pd.DataFrame()
    best_threshold = estimate_best_threshold(calib_vals)
    if hasattr(model, "threshold_rej"):
        print(best_threshold, model.threshold_rej)
    for target_coverage in target_coverages:
        # here we consider the case where there is a limited amount of instances that can be deferred to humans
        if (target_coverage is not None) and (target_coverage < 1):
            threshold = np.quantile(calib_vals["rej_score"], target_coverage)
            deferred = np.where(test_vals["rej_score"] > threshold, 1, 0)
        # here we consider the case where all instances can be deferred to humans
        # in this case the threshold is computed by linear search and maximizing the validation set accuracy
        # hence we select the point where we do not gain by deferring (we expect RDD to be not significant in this setting)
        elif target_coverage is None:
            threshold = best_threshold
            deferred = np.where(test_vals["rej_score"] > threshold, 1, 0)
        elif target_coverage == 1:
            threshold = np.max(calib_vals["rej_score"])
            deferred = np.zeros(len(test_vals["rej_score"]))
        correct = np.where(
            deferred == 1,
            test_vals["labels"] == test_vals["hum_preds"],
            test_vals["labels"] == test_vals["preds"],
        ).astype(float)
        # main()
        try:
            tmp, res_dict = get_rdd_robust_results(
                correct, test_vals["rej_score"], cutoff=threshold
            )
            print(rdrobust(correct, test_vals["rej_score"], c=threshold))
            # data_test (dict): dict data with fields 'defers', 'labels', 'hum_preds', 'preds'
            data_to_test = test_vals.copy()
            data_to_test["defers"] = deferred
            rs_metrics = compute_deferral_metrics(data_to_test)
            for key in rs_metrics.keys():
                tmp[key] = rs_metrics[key]
            tmp["threshold"] = threshold
            tmp["target_coverage"] = target_coverage
            if data == "synth":
                tmp[
                    "dataset"
                ] = "synth_{}_ExpertError{}_ExpertNDError{}_MachineError{}".format(
                    seed,
                    expert_deferred_error,
                    expert_nondeferred_error,
                    machine_nondeferred_error,
                )
            else:
                tmp["dataset"] = data
            tmp["method"] = model.__class__.__name__
            if data == "synth":
                tmp["data_distribution"] = data_distribution
                tmp["num_of_guassians"] = num_of_guassians
                tmp["d"] = d
                tmp["expert_deferred_error"] = expert_deferred_error
                tmp["expert_nondeferred_error"] = expert_nondeferred_error
                tmp["machine_nondeferred_error"] = machine_nondeferred_error
            res = pd.concat([res, tmp], axis=0)
            if os.path.exists("results/") is False:
                os.mkdir("results/")
            if os.path.exists("results/{}/".format(data)) is False:
                os.mkdir("results/{}/".format(data))
            if data != "synth":
                if print_time:
                    res["time_to_fit"] = time_to_fit
                    res.to_csv(
                        "results/{}/GCresultsWithTime_test_{}_{}_{}_ep{}.csv".format(
                            data, data, defer_system, seed, total_epochs
                        ),
                        index=False,
                    )
                else:
                    res.to_csv(
                        "results/{}/GCresults_test_{}_{}_{}_ep{}.csv".format(
                            data, data, defer_system, seed, total_epochs
                        ),
                        index=False,
                    )
            else:
                if print_time:
                    res["time_to_fit"] = time_to_fit
                    res.to_csv(
                        "results/{}/GCresultsWithTime_test_{}_{}_{}_{}_{}_{}_ep{}.csv".format(
                            data,
                            data,
                            defer_system,
                            seed,
                            expert_deferred_error,
                            expert_nondeferred_error,
                            machine_nondeferred_error,
                            total_epochs,
                        ),
                        index=False,
                    )
                else:
                    res.to_csv(
                        "results/{}/GCresults_test_{}_{}_{}_{}_{}_{}_ep{}.csv".format(
                            data,
                            data,
                            defer_system,
                            seed,
                            expert_deferred_error,
                            expert_nondeferred_error,
                            machine_nondeferred_error,
                            total_epochs,
                        ),
                        index=False,
                    )

        except:
            continue
    if data == "synth":
        tmp1 = get_raw_res(test_vals)
        tmp2 = get_raw_res(calib_vals)
        if os.path.exists("resultsRAW/") is False:
            os.mkdir("resultsRAW/")
        if os.path.exists("resultsRAW/{}/".format(data)) is False:
            os.mkdir("resultsRAW/{}/".format(data))
        tmp1.to_csv(
            "resultsRAW/{}/GCresultsRAW_test_{}_{}_{}_{}_{}_{}_ep{}.csv".format(
                data,
                data,
                defer_system,
                seed,
                expert_deferred_error,
                expert_nondeferred_error,
                machine_nondeferred_error,
                total_epochs,
            ),
            index=False,
        )
        tmp2.to_csv(
            "resultsRAW/{}/GCresultsRAW_cal_{}_{}_{}_{}_{}_{}_ep{}.csv".format(
                data,
                data,
                defer_system,
                seed,
                expert_deferred_error,
                expert_nondeferred_error,
                machine_nondeferred_error,
                total_epochs,
            ),
            index=False,
        )
    else:
        tmp1 = get_raw_res(test_vals)
        tmp2 = get_raw_res(calib_vals)
        if os.path.exists("resultsRAW/") is False:
            os.mkdir("resultsRAW/")
        if os.path.exists("resultsRAW/{}/".format(data)) is False:
            os.mkdir("resultsRAW/{}/".format(data))
        tmp1.to_csv(
            "resultsRAW/{}/GCresultsRAW_test_{}_{}_{}_ep{}.csv".format(
                data, data, defer_system, seed, total_epochs
            ),
            index=False,
        )
        tmp2.to_csv(
            "resultsRAW/{}/GCresultsRAW_cal_{}_{}_{}_ep{}.csv".format(
                data, data, defer_system, seed, total_epochs
            ),
            index=False,
        )


def main(all_models=False, all_data=False, **kwargs):
    if all_models:
        defer_systems = ["SP", "ASM", "LCE", "CC", "OVA", "DT", "RS"]
    else:
        defer_systems = [kwargs["defer_system"]]
    if all_data:
        data = ["synth", "hatespeech", "chestxray2", "galaxyzoo", "cifar10h"]
    else:
        data = [kwargs["data"]]
    kwargs.pop("data")
    kwargs.pop("defer_system")
    for d in tqdm(data):
        if d == "chestxray2":
            label_chosen = 2
        else:
            label_chosen = 0
        print(label_chosen)
        for ds in tqdm(defer_systems):
            test(data=d, defer_system=ds, label_chosen=label_chosen, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_distribution", type=str, default="mix_of_gaussians")
    parser.add_argument("--num_of_guassians", type=int, default=15)
    parser.add_argument("--d", type=int, default=30)
    parser.add_argument("--milp_time_limit", type=int, default=60 * 40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="synth")
    parser.add_argument("--defer_system", type=str, default="RS")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    # def main():
    args = parser.parse_args()
    seed = args.seed
    defer_system = args.defer_system
    data = args.data
    data_distribution = args.data_distribution
    expert_deferred_error = settings_synth["expert_deferred_error"]
    expert_nondeferred_error = settings_synth["expert_nondeferred_error"]
    machine_nondeferred_error = settings_synth["machine_nondeferred_error"]
    d = args.d
    total_epochs = args.epochs
    num_of_guassians = args.num_of_guassians
    device = args.device
    if data == "all":
        all_d = True
    else:
        all_d = False
    if defer_system == "all":
        all_m = True
    else:
        all_m = False
    main(
        all_data=all_d,
        all_models=all_m,
        seed=seed,
        defer_system=defer_system,
        data=data,
        data_distribution=data_distribution,
        expert_deferred_error=expert_deferred_error,
        expert_nondeferred_error=expert_nondeferred_error,
        machine_nondeferred_error=machine_nondeferred_error,
        d=d,
        num_of_guassians=num_of_guassians,
        device=device,
    )
