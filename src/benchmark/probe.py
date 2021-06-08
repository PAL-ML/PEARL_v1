import torch
import glob
import natsort
import re
import os
from torch import nn
from .utils import EarlyStopping, appendabledict, \
    calculate_multiclass_accuracy, calculate_multiclass_f1_score,\
    append_suffix, compute_dict_average

from copy import deepcopy
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .categorization import summary_key_dict


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.model = nn.Linear(in_features=input_dim, out_features=num_classes)

    def forward(self, feature_vectors):
        return self.model(feature_vectors)

class NonLinearProbe1(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, feature_vectors):
        return self.relu(self.linear(feature_vectors))

class NonLinearProbe2(nn.Module):
    def __init__(self, input_dim, num_hidden=300, num_classes=255):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=num_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=num_hidden, out_features=num_classes)

    def forward(self, feature_vectors):
        x = self.linear1(feature_vectors)
        x = self.relu(x)
        return self.linear2(x)

class NonLinearProbe3(nn.Module):
    def __init__(self, input_dim, num_classes=255):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_vectors):
        return self.sigmoid(self.linear(feature_vectors))

class NonLinearProbe4(nn.Module):
    def __init__(self, input_dim, num_hidden=300, num_classes=255):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=num_hidden)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=num_hidden, out_features=num_hidden)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=num_hidden, out_features=num_classes)

    def forward(self, feature_vectors):
        x = self.linear1(feature_vectors)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return self.linear3(x)

class NonLinearProbe5(nn.Module):
    def __init__(self, input_dim, num_hidden=300, num_classes=255):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=num_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=num_hidden, out_features=num_classes)
        self.bn = nn.BatchNorm1d(num_hidden)

    def forward(self, feature_vectors):
        x = self.linear1(feature_vectors)
        x = self.relu(x)
        x = self.bn(x)
        return self.linear2(x)


class LstmProbe(nn.Module):
    def __init__(self, input_dim, n_layers=2, n_hidden=300, num_classes=255):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, n_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, num_classes)

    def forward(self, feature_vectors):
        out =  self.lstm(feature_vectors)
        return self.linear(out)

class LstmProbe2(nn.Module):
    def __init__(self, input_dim, n_layers=3, n_hidden=1024, num_classes=255):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, n_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, num_classes)

    def forward(self, feature_vectors):
        out =  self.lstm(feature_vectors)
        return self.linear(out)
      
class LstmProbe3(nn.Module):
    def __init__(self, input_dim, n_layers=6, n_hidden=2048, num_classes=255):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, n_hidden, n_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, num_classes)

    def forward(self, feature_vectors):
        out =  self.lstm(feature_vectors)
        return self.linear(out)

class FullySupervisedLinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=255):
        super().__init__()
        self.encoder = deepcopy(encoder)
        self.probe = LinearProbe(input_dim=self.encoder.hidden_size,
                                 num_classes=num_classes)

    def forward(self, x):
        feature_vec = self.encoder(x)
        return self.probe(feature_vec)


class ProbeTrainer():
    def __init__(self,
                 encoder=None,
                 method_name="my_method",
                 probe_type="linear",
                 wandb=None,
                 patience=15,
                 num_classes=256,
                 fully_supervised=False,
                 save_dir=".models",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 lr=5e-4,
                 epochs=100,
                 batch_size=64,
                 representation_len=256):

        self.encoder = encoder
        self.wandb = wandb
        self.device = device
        self.fully_supervised = fully_supervised
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.method = method_name
        self.feature_size = representation_len
        self.loss_fn = nn.CrossEntropyLoss() 
        self.valid_probe_types = set(['linear', 'lstm', 'non-linear-1', 'non-linear-2', 'non-linear-3', 'non-linear-4'])

        if not self.is_probe_type_valid(probe_type):
            raise Exception("Invalid probe type. Pick amongst ")

        self.probe_type = probe_type

        # bad convention, but these get set in "create_probes"
        self.probes = self.early_stoppers = self.optimizers = self.schedulers = None
        
        # initialized in load checkpoints 
        self.loaded_model_paths = None

    def is_probe_type_valid(self, probe_type, ):
        if probe_type in self.valid_probe_types:
            return True
        else:
            return False
    
    def _init_probes(self, sample_label):
        
        if self.fully_supervised:
            assert self.encoder != None, "for fully supervised you must provide an encoder!"
            self.probes = {k: FullySupervisedLinearProbe(encoder=self.encoder,
                                                         num_classes=self.num_classes).to(self.device) for k in
                           sample_label.keys()}

        elif self.probe_type=='linear':
            self.probes = {k: LinearProbe(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

            self.load_probe_checkpoints(self.save_dir, to_train=True)
            
        elif self.probe_type=='lstm':
            self.probes = {k: LstmProbe(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}
            self.load_probe_checkpoints(self.save_dir, to_train=True)
            
        elif self.probe_type=='lstm-2':
            self.probes = {k: LstmProbe2(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

            self.load_probe_checkpoints(self.save_dir, to_train=True)
            
        elif self.probe_type=='lstm-3':
            self.probes = {k: LstmProbe3(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}            

            self.load_probe_checkpoints(self.save_dir, to_train=True)
            
        elif self.probe_type=='non-linear-1':
            self.probes = {k: NonLinearProbe1(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

            self.load_probe_checkpoints(self.save_dir, to_train=True)
        
        elif self.probe_type=='non-linear-2':
            self.probes = {k: NonLinearProbe2(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

            self.load_probe_checkpoints(self.save_dir, to_train=True)

        elif self.probe_type=='non-linear-3':
            self.probes = {k: NonLinearProbe3(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

            self.load_probe_checkpoints(self.save_dir, to_train=True)
        
        elif self.probe_type=='non-linear-4':
            self.probes = {k: NonLinearProbe4(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

            self.load_probe_checkpoints(self.save_dir, to_train=True)
        
        elif self.probe_type=='non-linear-5':
            self.probes = {k: NonLinearProbe5(input_dim=self.feature_size,
                                          num_classes=self.num_classes).to(self.device) for k in sample_label.keys()}

            self.load_probe_checkpoints(self.save_dir, to_train=True)


    def create_probes(self, sample_label):
       
        self._init_probes(sample_label)

        self.early_stoppers = {
            k: EarlyStopping(patience=self.patience, verbose=False, name=k + "_probe", save_dir=self.save_dir)
            for k in sample_label.keys()}

        self.optimizers = {k: torch.optim.Adam(list(self.probes[k].parameters()),
                                               eps=1e-5, lr=self.lr) for k in sample_label.keys()}
        self.schedulers = {
            k: torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[k], patience=5, factor=0.2, verbose=True,
                                                          mode='max', min_lr=1e-5) for k in sample_label.keys()}

    def generate_batch(self, episodes, episode_labels):
        total_steps = sum([len(e) for e in episodes])
        assert total_steps > self.batch_size
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)

        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            episode_labels_batch = [episode_labels[x] for x in indices]
            xs, labels = [], appendabledict()
            for ep_ind, episode in enumerate(episodes_batch):
                # Get one sample from this episode
                t = np.random.randint(len(episode))
                xs.append(episode[t])
                labels.append_update(episode_labels_batch[ep_ind][t])
            yield torch.stack(xs).float().to(self.device) / 255., labels

    def probe(self, batch, k):
        probe = self.probes[k]
        probe.to(self.device)
        if self.fully_supervised:
            # if method is supervised batch is a batch of frames and probe is a full encoder + linear or nonlinear probe
            preds = probe(batch)

        elif not self.encoder:
            # if encoder is None then inputs are vectors
            f = batch.detach()
            assert len(f.squeeze().shape) == 2, "if input is not a batch of vectors you must specify an encoder!"
            preds = probe(f)

        else:
            with torch.no_grad():
                self.encoder.to(self.device)
                f = self.encoder(batch).detach()
            preds = probe(f)
        return preds

    def do_one_epoch(self, episodes, label_dicts):
        sample_label = label_dicts[0][0]
        epoch_loss, accuracy = {k + "_loss": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}, \
                               {k + "_acc": [] for k in sample_label.keys() if
                                not self.early_stoppers[k].early_stop}

        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                if self.early_stoppers[k].early_stop:
                    continue
                optim = self.optimizers[k]
                optim.zero_grad()

                label = torch.tensor(label).long().to(self.device)
                preds = self.probe(x, k)

                loss = self.loss_fn(preds, label)

                epoch_loss[k + "_loss"].append(loss.detach().item())
                preds = preds.cpu().detach().numpy()
                preds = np.argmax(preds, axis=1)
                label = label.cpu().detach().numpy()
                accuracy[k + "_acc"].append(calculate_multiclass_accuracy(preds,
                                                                          label))
                if self.probes[k].training:
                    loss.backward()
                    optim.step()

        epoch_loss = {k: np.mean(loss) for k, loss in epoch_loss.items()}
        accuracy = {k: np.mean(acc) for k, acc in accuracy.items()}

        return epoch_loss, accuracy

    def do_test_epoch(self, episodes, label_dicts):
        sample_label = label_dicts[0][0]
        accuracy_dict, f1_score_dict = {}, {}
        pred_dict, all_label_dict = {k: [] for k in sample_label.keys()}, \
                                    {k: [] for k in sample_label.keys()}

        data_generator = self.generate_batch(episodes, label_dicts)
        for step, (x, labels_batch) in enumerate(data_generator):
            for k, label in labels_batch.items():
                label = torch.tensor(label).long().cpu()
                all_label_dict[k].append(label)
                preds = self.probe(x, k).detach().cpu()
                pred_dict[k].append(preds)

        for k in all_label_dict.keys():
            preds, labels = torch.cat(pred_dict[k]).cpu().detach().numpy(),\
                            torch.cat(all_label_dict[k]).cpu().detach().numpy()

            preds = np.argmax(preds, axis=1)
            accuracy = calculate_multiclass_accuracy(preds, labels)
            f1score = calculate_multiclass_f1_score(preds, labels)
            accuracy_dict[k] = accuracy
            f1_score_dict[k] = f1score

        return accuracy_dict, f1_score_dict
    
    def save_checkpoint(self, name, model, num_epochs=None):
        '''Saves model'''
        
        if num_epochs:
            filepath = str(self.save_dir) + "/" + str(name) + '_' + str(num_epochs) + ".pt"
        else:
            filepath = str(self.save_dir) + "/" + str(name) + '_' + "final" + ".pt"

        torch.save(model.state_dict(), filepath)
    
    def load_checkpoint(self, model_path, to_train=True, cls=LinearProbe):
        '''Loads model'''
        model = cls(input_dim=self.feature_size, num_classes=self.num_classes)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

        if to_train:
            model.train()
        else:
            model.eval()
        
        return model

    def save_probe_checkpoints(self, epochs=None):
        for k, probe in self.probes.items():
            self.save_checkpoint(k, probe, num_epochs=epochs)

        print("All probe checkpoints saved!")

    def load_probe_checkpoints(self, path, cls=LinearProbe, to_train=False, log=True):
        path = os.path.join(path, "*") # get all files in folder
        all_files = glob.glob(path)
        self.loaded_model_paths = {}

        for k in self.probes.keys():
            #probe_specifc_files = list(filter(lambda x: k in ''.join(x.split('_')[:-1]), all_files))
            #selected_file = list(filter(lambda x: "final" in ''.join(x.split('_')[:-1]), all_files))
            probe_specifc_files = list(filter(lambda x: k in ''.join(x.split('/')[-1]), all_files))
            selected_file = list(filter(lambda x: "final" in ''.join(x.split('/')[-1]), all_files))
            if len(selected_file) == 0:
                sorted_list = natsort.natsorted(probe_specifc_files)
                if len(probe_specifc_files) > 0:
                    selected_file = [sorted_list[-1]]
                else:
                    selected_file = []
            
            model_path = selected_file[0] if len(selected_file) > 0 else None
            self.loaded_model_paths[k] = model_path

            if model_path:
                self.probes[k] = self.load_checkpoint(model_path, to_train=to_train, cls=cls)
        
        if log:
            for k, loaded_path in self.loaded_model_paths.items():
                print("K: {}, Loaded path: {}".format(k, loaded_path))

    def get_num_epochs_trained(self):
        if not self.loaded_model_paths or len(self.loaded_model_paths.keys()) == 0:
            print("returned 0 epochs trained 1") # folder doest exist for env to be trained
            return 0
        
        # Assumes all models are trained simultaneously 
        for k, model_path in self.loaded_model_paths.items():
            if model_path: # if model path is not none
                file_name = model_path.split("/")[-1] # ignore rest og filepath
                int_list = re.findall(r'\d+', file_name)
                #print("model_path: {}".format(model_path))
                #print("ints in model file path: {}".format(int_list))
                if len(int_list)== 0:
                    if 'final' in file_name:
                        print("final found in model path name")
                        return -1  # return special value if model loaded has final tag
                    else: 
                        print("no model path found. Hence to prior training")
                        return 0
                else:
                    num_epochs = int(int_list[0]) # assumes first number that exists in the model filepath is num epochs
                    print("selected num_epochs: {}".format(num_epochs))
                return num_epochs
        print("returned 0 epochs trained 2")  # folder exists but no trained models
        return 0    

        
    def train(self, tr_eps, val_eps, tr_labels, val_labels, save_interval=5):
        # if not self.encoder:
        #     assert len(tr_eps[0][0].squeeze().shape) == 2, "if input is a batch of vectors you must specify an encoder!"
        sample_label = tr_labels[0][0]
        self.create_probes(sample_label)

        num_epochs_trained = self.get_num_epochs_trained()

        if ((self.epochs - num_epochs_trained) <= 0) or (num_epochs_trained == -1):
            print("Alread trained to {} epochs.".format(num_epochs_trained))
            e = self.epochs
            print("Probes have already been trained, but are trying to be trained again...")
        elif (self.epochs - num_epochs_trained) > 0:
            e = num_epochs_trained + 1
        else:
            e = 0

        all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        while (not all_probes_stopped) and e < self.epochs:
            epoch_loss, accuracy = self.do_one_epoch(tr_eps, tr_labels)
            self.log_results(e, epoch_loss, accuracy)

            val_loss, val_accuracy = self.evaluate(val_eps, val_labels, epoch=e)
            # update all early stoppers
            for k in sample_label.keys():
                if not self.early_stoppers[k].early_stop:
                    self.early_stoppers[k](val_accuracy["val_" + k + "_acc"], self.probes[k])

            for k, scheduler in self.schedulers.items():
                if not self.early_stoppers[k].early_stop:
                    scheduler.step(val_accuracy['val_' + k + '_acc'])
            e += 1
            if e % save_interval == 0 and e != 0:
                self.save_probe_checkpoints(epochs=e)
            all_probes_stopped = np.all([early_stopper.early_stop for early_stopper in self.early_stoppers.values()])
        print("All probes early stopped!")

    def evaluate(self, val_episodes, val_label_dicts, epoch=None):
        for k, probe in self.probes.items():
            probe.eval()
        epoch_loss, accuracy = self.do_one_epoch(val_episodes, val_label_dicts)
        epoch_loss = {"val_" + k: v for k, v in epoch_loss.items()}
        accuracy = {"val_" + k: v for k, v in accuracy.items()}
        self.log_results(epoch, epoch_loss, accuracy)
        for k, probe in self.probes.items():
            probe.train()
        return epoch_loss, accuracy

    def test(self, test_episodes, test_label_dicts, epoch=None):
        for k in self.early_stoppers.keys():
            self.early_stoppers[k].early_stop = False
        for k, probe in self.probes.items():
            probe.eval()
        acc_dict, f1_dict = self.do_test_epoch(test_episodes, test_label_dicts)

        acc_dict, f1_dict = postprocess_raw_metrics(acc_dict, f1_dict)

        print("""In our paper, we report F1 scores and accuracies averaged across each category. 
              That is, we take a mean across all state variables in a category to get the average score for that category.
              Then we average all the category averages to get the final score that we report per game for each method. 
              These scores are called \'across_categories_avg_acc\' and \'across_categories_avg_f1\' respectively
              We do this to prevent categories with large number of state variables dominating the mean F1 score.
              """)
        self.log_results("Test", acc_dict, f1_dict)
        return acc_dict, f1_dict
    
    def log_results(self, epoch_idx, *dictionaries):
        print("Epoch: {}".format(epoch_idx))
        self.log_wandb_results(epoch_idx, dictionaries)
        for dictionary in dictionaries:
            for k, v in dictionary.items():
                print("\t {}: {:8.4f}".format(k, v))
            print("\t --")
    
    def log_wandb_results(self, epoch_idx, *dictionaries):

        if type(epoch_idx) == str:
            for dictionary in dictionaries:
                dict = dictionary[0]
                self.wandb.log(dict)
                self.wandb.log(dictionary[1])
        else:
            for dictionary in dictionaries:
                dict = dictionary[0]
                self.wandb.log(dict, step=epoch_idx)
                self.wandb.log(dictionary[1], step=epoch_idx)

def postprocess_raw_metrics(acc_dict, f1_dict):
    acc_overall_avg, f1_overall_avg = compute_dict_average(acc_dict), \
                                      compute_dict_average(f1_dict)
    acc_category_avgs_dict, f1_category_avgs_dict = compute_category_avgs(acc_dict), \
                                                    compute_category_avgs(f1_dict)
    acc_avg_across_categories, f1_avg_across_categories = compute_dict_average(acc_category_avgs_dict), \
                                                          compute_dict_average(f1_category_avgs_dict)
    acc_dict.update(acc_category_avgs_dict)
    f1_dict.update(f1_category_avgs_dict)

    acc_dict["overall_avg"], f1_dict["overall_avg"] = acc_overall_avg, f1_overall_avg
    acc_dict["across_categories_avg"], f1_dict["across_categories_avg"] = [acc_avg_across_categories,
                                                                           f1_avg_across_categories]

    acc_dict = append_suffix(acc_dict, "_acc")
    f1_dict = append_suffix(f1_dict, "_f1")

    return acc_dict, f1_dict


def compute_category_avgs(metric_dict):
    category_dict = {}
    for category_name, category_keys in summary_key_dict.items():
        category_values = [v for k, v in metric_dict.items() if k in category_keys]
        if len(category_values) < 1:
            continue
        category_mean = np.mean(category_values)
        category_dict[category_name + "_avg"] = category_mean
    return category_dict



