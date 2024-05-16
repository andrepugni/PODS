data_distribution = "mix_of_gaussians"
expert_deferred_error = 0.0
expert_nondeferred_error = 0.30
machine_nondeferred_error = 0
num_of_guassians = 15
d = 30
seed = 42
data = "synth"
defer_system = "RS"
total_epochs = 100
device = "cuda:0"
label_chosen = 0
num_classes_dict = {
    "synth": 2,
    "chestxray0": 2,
    "chestxray1": 2,
    "hatespeech": 3,
    "cifar10h": 10,
}
filename = "synth"


settings_synth = {
    "data_distribution": "mix_of_gaussians",
    "expert_deferred_error": 0.1,
    "expert_nondeferred_error": 0.30,
    "machine_nondeferred_error": 0.2,
    "num_of_guassians": 15,
    "d": 30,
    "seed": 42,
    "data": "synth",
    "defer_system": "RS",
    "total_epochs": 100,
    "n_classes": 2,
}
