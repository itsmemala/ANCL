import torch
import itertools
from argparse import ArgumentParser
from copy import deepcopy
import pickle
import os
import io

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class Appr(Inc_Learning_Appr):
    """Class implementing LA-MAS approach"""
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=[50], lamb_up_max =[1000.0], lamb_up_mult=[1.0], lamb_down=[1.0], tau_alpha1=[0.8], tau_alpha2=[0.0],
                 save_alphas=False, alpha_rel_save_path='',
                 save_models_imps_path=None, retrain_main_task_model=None,
                 alpha=0.5, 
                 fi_num_samples=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.lamb = lamb
        self.lamb_up_max = lamb_up_max
        self.lamb_up_mult = lamb_up_mult
        self.lamb_down = lamb_down
        self.tau_alpha1 = tau_alpha1
        self.tau_alpha2 = tau_alpha2
        self.save_alphas = save_alphas
        self.alpha_rel_save_path = alpha_rel_save_path
        self.save_models_imps_path = save_models_imps_path
        self.retrain_main_task_model = retrain_main_task_model
        self.alpha = alpha
        self.num_samples = fi_num_samples
        self.model_aux = None
        self.optimizer_expand = None

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store importance
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                           if p.requires_grad}
        # Parameter of auxiliary network
        self.auxiliary_params = None
        # Store importance for auxiliary network
        self.importance_aux = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                       if p.requires_grad}

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # lambda is the regularizer trade-off -- In original code: MAS.ipynb block [4]: lambda set to 1
        parser.add_argument('--lamb', default=[50], type=list_of_floats, required=False,
                            help='Forgetting-intransigence trade-off  (default=%(default)s)')
        parser.add_argument('--lamb_up_max', default=[1.0], type=list_of_floats, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--lamb_up_mult', default=[1.0], type=list_of_floats, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--lamb_down', default=[1.0], type=list_of_floats, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--tau_alpha1', default=[0.8], type=list_of_floats, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--tau_alpha2', default=[0.0], type=list_of_floats, required=False,
                                    help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--save_alphas', default=False, type=bool, required=False,
                            help='Whether to save computed alphas (default=%(default)s)')
        parser.add_argument('--alpha_rel_save_path', default='', type=str, required=False,
                                    help='Path to save computed alphas (default=%(default)s)')
        parser.add_argument('--save_models_imps_path', default=None, type=str, required=False,
                                    help='Path to save models and importances for training from middle (default=%(default)s)')
        parser.add_argument('--retrain_main_task_model', default=None, type=int, required=False,
                                    help='Which task to re-train main network (for hyp search) (default=%(default)s)')
        # lambda_e sets how important the new task is compared to the old one
        # parser.add_argument('--lamb-a', default=5, type=float, required=False,
        #                     help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Define how old and new importance is fused, by default it is a 50-50 fusion
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='A-MAS alpha (default=%(default)s)')
        # Number of samples from train for estimating importance
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for MAS Importance (-1: all available) (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    #= MAS (global) is implemented since the paper shows is more efficient than l-MAS (local)
    def estimate_parameter_importance(self, model, trn_loader):
        # Initialize importance matrices
        importance = {n: torch.zeros(p.shape).to(self.device) for n, p in model.model.named_parameters()
                      if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            model.zero_grad()
            # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
            outputs = model.forward(images.to(self.device))
            # labels not required, "...use the gradients of the squared L2-norm of the learned function output."
            loss = torch.norm(torch.cat(outputs, dim=1), p=2, dim=1).mean()
            loss.backward()
            # accumulate the gradients over the inputs to obtain importance weights
            for n, p in model.model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(targets)
        # divide by N total number of samples
        n_samples = n_samples_batches * trn_loader.batch_size
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance
    
    def compute_la_importance(self,t,fisher_old,fisher,lamb,lamb_up_max,lamb_up_mult,lamb_down,tau_alpha1,tau_alpha2):
        modified_fisher = {}
        fisher_rel_dict = {}
        for n in fisher.keys():
            modified_fisher[n] = fisher_old[n]
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]+0.0000000001) # Relative importance
            fisher_rel_dict[n] = fisher_rel
            # frel_cut = torch.nan_to_num(torch.mean(fisher_rel.flatten())).item()
            frel_mn = torch.mean(fisher_rel.flatten())
            frel_std = torch.std(fisher_rel.flatten())
            frel_cut = tau_alpha1*frel_mn + (tau_alpha2*frel_std)
            lamb_up_min = torch.ceil(1/max(frel_cut,0.05))
            if lamb_up_max is not None:
                lamb_up_max = max(lamb_up_max/lamb,lamb_up_min)
                lamb_up = lamb_up_min + lamb_up_mult*(lamb_up_max-lamb_up_min)
                assert lamb_up <= lamb_up_max
            if lamb_up is not None: assert lamb_up >= lamb_up_min
            # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
            modified_fisher[n][fisher_rel>frel_cut] = lamb_up*fisher_rel[fisher_rel>frel_cut]*fisher_old[n][fisher_rel>frel_cut]
            # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
            modified_fisher[n][fisher_rel<=frel_cut] = lamb_down*fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
        if self.save_alphas:
            with open(self.alpha_rel_save_path+'/t'+str(t)+'_fisher_rel.pkl', 'wb') as fp:
                pickle.dump(fisher_rel_dict, fp)
        return modified_fisher

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # Load model if already trained (except when doing hyp-param search):
        main_model_path = self.save_models_imps_path+"_t"+str(t)+"_model_state_dict"
        if t<self.retrain_main_task_model and os.path.exists(main_model_path):
            self.model.load_state_dict(torch.load(main_model_path))
            print('=' * 108)
            print("Loaded Main Network. No Training.")
            print('=' * 108)
        else:
            # add exemplars to train_loader
            if len(self.exemplars_dataset) > 0 and t > 0:
                trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                        batch_size=trn_loader.batch_size,
                                                        shuffle=True,
                                                        num_workers=trn_loader.num_workers,
                                                        pin_memory=trn_loader.pin_memory)
            print("lamb : ", self.lamb[t])
            if t > 0:
                # Load model if already trained (except when doing hyp-param search):
                aux_model_path = self.save_models_imps_path+"_t"+str(t)+"_aux_model_state_dict"
                la_imp_path = self.save_models_imps_path+"_t"+str(t)+"_la_imp.pkl"
                if os.path.exists(aux_model_path):
                    self.model_aux = deepcopy(self.model)
                    self.model_aux.load_state_dict(torch.load(aux_model_path))
                    with open(la_imp_path, 'rb') as handle:
                        self.importance_aux = CPU_Unpickler(handle).load()
                    print('=' * 108)
                    print("Loaded Aux Network and LA imp. No Training.")
                    print('=' * 108)
                else:
                    print('=' * 108)
                    print("Training of Auxiliary Network")
                    print('=' * 108)
                    # Args for the new trainer
                    new_trainer_args = dict(nepochs=self.nepochs, lr=self.lr, lr_min=self.lr_min, lr_factor=self.lr_factor,
                                    lr_patience=self.lr_patience, clipgrad=self.clipgrad, momentum=0.9,
                                    wd=5e-4, multi_softmax=self.multi_softmax, wu_nepochs=self.warmup_epochs,
                                    wu_lr_factor=self.warmup_lr, fix_bn=self.fix_bn, logger=self.logger)
                    self.model_aux = deepcopy(self.model)
                    # Train auxiliary model on current dataset
                    new_trainer = NewTaskTrainer(self.model_aux, self.device, **new_trainer_args)
                    new_trainer.train_loop(t, trn_loader, val_loader)

                    # Store parameter of auxiliary model to compute regularizer later
                    self.auxiliary_params = {n: p.clone().detach() for n, p in self.model_aux.model.named_parameters() if p.requires_grad}

                    # calculate importance of auxiliary model -> then compute la importance
                    curr_importance = self.estimate_parameter_importance(self.model_aux, trn_loader)
                    la_importance = self.compute_la_importance(t,self.importance,curr_importance,self.lamb[t],self.lamb_up_max[t],self.lamb_up_mult[t],self.lamb_down[t],self.tau_alpha1[t],self.tau_alpha2[t])
                    
                    for n in self.importance_aux.keys():
                        # self.importance_aux[n] = curr_importance[n]
                        self.importance_aux[n] = la_importance[n]

                    # save aux network and la imp to re-use and avoid training each time during hyp-param search
                    torch.save(self.model_aux.state_dict(), aux_model_path)
                    with open(la_imp_path, 'wb') as fp:
                        pickle.dump(la_importance, fp)

            print('=' * 108)
            print("Training of Main Network")
            print('=' * 108)
            # FINETUNING TRAINING -- contains the epochs loop
            super().train_loop(t, trn_loader, val_loader)

            # EXEMPLAR MANAGEMENT -- select training subset
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

            # Save model to re-use and avoid training each time during hyp-param search
            torch.save(self.model.state_dict(), main_model_path)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

        # calculate importance
        curr_importance = self.estimate_parameter_importance(self.model, trn_loader)
        # merge importance, we do not want to keep importance for each task in memory
        for n in self.importance.keys():
            # Added option to accumulate importance over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                self.importance[n] = alpha * self.importance[n] + (1 - alpha) * curr_importance[n]
            else:
                # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
                self.importance[n] = self.alpha * self.importance[n] + (1 - self.alpha) * curr_importance[n]

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # loss_reg = 0
            # # memory aware synapses regularizer penalty
            # for n, p in self.model.model.named_parameters():
            #     if n in self.importance.keys():
            #         loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2)) / 2
            loss_reg_exp = 0
            # la penalty
            for n, p in self.model.model.named_parameters():
                if n in self.importance_aux.keys():
                    loss_reg_exp += torch.sum(self.importance_aux[n] * (p - self.auxiliary_params[n]).pow(2)) / 2            
            # loss += self.lamb * loss_reg + self.lamb_a * loss_reg_exp
            loss += self.lamb[t] * loss_reg_exp
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

class NewTaskTrainer(Inc_Learning_Appr):
    def __init__(self, model, device, nepochs=160, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None):
        super(NewTaskTrainer, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                                             momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn,
                                             eval_on_train, logger)
