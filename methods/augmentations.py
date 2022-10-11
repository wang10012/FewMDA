from regex import D
import torch
import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as nnf
import torch.nn as nn
import torch.backends.cudnn as cudnn
from transformers import AdamW
import torchvision
# from helpers.clip_transformations import EmbeddingDataset
import wandb
import open_clip
from scipy.spatial import distance

import clip
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from scipy import stats
import uuid
try:
    from progress_bar import progress_bar
except:
    progress_bar = lambda current, total, msg: None

import helpers.data_helpers as dh
from methods.clip_transformations import EmbeddingDataset
from clip_utils import *
from methods import predictors

import omegaconf
from omegaconf import OmegaConf

class Augment:
    """
    Class for augmenting clip embeddings in embedding space.
    """

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        self.cfg = cfg
        torch.manual_seed(cfg.EXP.SEED)
        self.image_features = image_features
        self.labels = labels
        self.domain_labels = domain_labels
        self.group_labels = group_labels
        self.text_features = text_features
        self.filenames = filenames
        self.class_names = dh.DATASET_CLASSES[self.cfg.DATA.DATASET]
        self.domain_names = dh.DATASET_DOMAINS[self.cfg.DATA.DATASET]
        self.domain_indexes = []
        self.alpha = self.cfg.AUGMENTATION.ALPHA
        self.include_orig = self.cfg.AUGMENTATION.INCLUDE_ORIG_TRAINING
        self.prompts = list(self.cfg.EXP.TEXT_PROMPTS)
        self.neutral_prompts = list(self.cfg.EXP.NEUTRAL_TEXT_PROMPTS)
        for prompt in list(self.cfg.EXP.TEXT_PROMPTS):
            if len(list(self.cfg.AUGMENTATION.DOM_LABELS)) == 0:
                if type(prompt) == omegaconf.listconfig.ListConfig:
                    print("remove list")
                    prompt = prompt[0]
                print(type(prompt), prompt, self.domain_names)
                dom_idx = [i for i in range(len(self.domain_names)) if self.domain_names[i] in prompt]
                assert len(dom_idx) == 1, "error indexing domains, make sure your text prompt contains the name of the domain"
                self.domain_indexes.append(dom_idx[0])
            else:
                self.domain_indexes = [self.domain_names.index(p) for p in list(self.cfg.AUGMENTATION.DOM_LABELS)]
        print("domain indexes ", self.domain_indexes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.cfg.EXP.IMAGE_FEATURES == 'clip':
            model, preprocess = clip.load(self.cfg.EXP.CLIP_MODEL, device)
        elif self.cfg.EXP.IMAGE_FEATURES == 'openclip':
            model, _, preprocess = open_clip.create_model_and_transforms(self.cfg.EXP.CLIP_MODEL, pretrained=self.cfg.EXP.CLIP_PRETRAINED_DATASET)
            model = model.to(torch.device('cuda:1'))
        model.eval()
        self.model = model

    def get_orig_text_embeddings(self, prompts):
        if self.cfg.AUGMENTATION.GENERIC:
            text_embs = zeroshot_classifier(prompts, self.model, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy().T
            text_embs /= np.linalg.norm(text_embs, axis=-1, keepdims=True)
            return text_embs
        else:
            all_texts, text_names = [], []
            for t in prompts:
                texts = [[t.format(c)] for c in self.class_names]
                text_names += texts
                text_emb = zeroshot_classifier(texts, self.model, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy().T
                text_emb /= np.linalg.norm(text_emb, axis=-1, keepdims=True)
                all_texts.append(text_emb)
            # print("array of orig prompts ", text_names, np.array(all_texts).transpose((1, 0, 2)).shape)
            return np.array(all_texts).transpose((1, 0, 2))

    def augment_single(self, img_embedding, label):
        """
        Augments a single image embedding with the text embedding.
        """
        # img_embedding /= np.linalg.norm(img_embedding, axis=-1, keepdims=True)
        return img_embedding

    def augment_dataset(self):
            """
            Augments the dataset
            """
            augmented_features = []
            augmented_labels = []
            augmented_domain_labels = []
            augmented_group_labels = []
            augmented_filenames = []
            offset = 1 if self.cfg.AUGMENTATION.INCLUDE_ORIG_TRAINING else 0
            for i, feature in enumerate(self.image_features):
                augmented_features += self.augment_single(feature, self.labels[i])
                augmented_labels += [self.labels[i] for _ in range(len(self.domain_indexes)+offset)]
                if offset == 1:
                    augmented_domain_labels += [self.domain_labels[i]] + [j for j in self.domain_indexes]
                else:
                    augmented_domain_labels += [j for j in self.domain_indexes]
                augmented_group_labels += [self.group_labels[i] for _ in range(len(self.domain_indexes)+offset)]
                augmented_filenames += [self.filenames[i] for _ in range(len(self.domain_indexes)+offset)]
            return np.array(augmented_features), np.array(augmented_labels), np.array(augmented_domain_labels), np.array(augmented_group_labels), np.array(augmented_filenames)

class Addition(Augment):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)
        self.class_names = dh.DATASET_CLASSES[cfg.DATA.DATASET]
        self.orig_text_embs = self.get_orig_text_embeddings(self.prompts)
        self.neutral_embedding = zeroshot_classifier([self.neutral_prompts], self.model, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy().T[0]
        self.neutral_embedding /= np.linalg.norm(self.neutral_embedding, axis=-1, keepdims=True)
        print("neutral embedding size ", self.neutral_embedding.shape)
        

    def get_interp(self, img_embedding, text_features, orig_text_embeddings):
        if self.alpha == 'cosine':
            img_embedding /= np.linalg.norm(img_embedding, axis=-1, keepdims=True)
            source_cs = np.abs(np.dot(img_embedding, self.neutral_embedding))
            print(np.linalg.norm(orig_text_embeddings[0]), np.linalg.norm(img_embedding), np.linalg.norm(self.neutral_embedding))
            print(source_cs)
            print([np.abs(np.dot(img_embedding, t)) for t in orig_text_embeddings])
            alphas = [np.exp(source_cs) / (np.exp(source_cs) + np.exp(np.abs(np.dot(img_embedding, t)))) for t in orig_text_embeddings]
            print(alphas)
            wandb.log({'alphas': alphas[0]})
        else:
            alphas = [self.alpha for t in text_features]
        return [(1 - a) * img_embedding + a * feat for a, feat in zip(alphas, text_features)]

    def augment_single(self, img_embedding, label):
        """
        Augments a single by adding the text emb for each domain to the given img emb and renormalizing 
        """
        img_embedding = super().augment_single(img_embedding, label)
        if self.cfg.AUGMENTATION.GENERIC:
            aug_embedding = np.array([img_embedding] + self.get_interp(img_embedding, self.text_features, self.orig_text_embs))
        else:
            aug_embedding = np.array([img_embedding] + self.get_interp(img_embedding, self.text_features[int(label)], self.orig_text_embs[int(label)]))
        aug_embedding /= np.linalg.norm(aug_embedding, axis=-1, keepdims=True)
        return list(aug_embedding)

class AdditionAlpha(Addition):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)

    def get_interp(self, img_embedding, text_features):
        """
        Augments a single by adding the text emb for each domain multiplied by a random a in (0,1) 
        to the given img emb and renormalizing 
        """
        return [(1 - self.alpha) * img_embedding + self.alpha * random.random() * feat for feat in text_features]

class Projection(Addition):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)

    def get_interp(self, img_embedding, text_features):
        """
        Augments a single by adding the projection of the img emb onto the text emb to the given img emb and renormalizing 
        """
        return [(1 - self.alpha) * img_embedding + self.alpha * projection(feat, img_embedding) for feat in text_features]

class OrthogonalProjection(Augment):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)

    def get_interp(self, img_embedding, text_features):
        """
        Augments a single by subtracting the projection of the img emb onto the text emb to the given img emb and renormalizing 
        """
        return [(1 - self.alpha) * img_embedding - self.alpha * projection(feat, img_embedding) for feat in text_features]

class Average(Augment):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)

    def augment_single(self, img_embedding, label):
        """
        Augments a single by averaging the text emb for each domain with the given img emb and renormalizing 
        """
        img_embedding = super().augment_single(img_embedding, label)
        if self.cfg.AUGMENTATION.GENERIC:
            aug_embedding = np.array([img_embedding] + [(img_embedding + feat)/2 for feat in self.text_features])
        else:
            aug_embedding = np.array([img_embedding] + [(img_embedding + feat)/2 for feat in self.text_features[int(label)]])
        aug_embedding /= np.linalg.norm(aug_embedding, axis=-1, keepdims=True)
        return list(aug_embedding)

class SLERP(Addition):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)
        # assumes text features is a list 
        
    def get_interp(self, img_embedding, text_features, orig_text_embeddings):
        """
        Augments a single by taking the shperical interpolation of the image emb and text emn
        """
        return [self.slerp(img_embedding, feat, self.alpha) for feat in text_features]

    def get_text_embeddings(self, model_name, prompts, norm=True):
        text_embs = zeroshot_classifier(prompts, self.model, model_type=self.cfg.EXP.IMAGE_FEATURES).cpu().numpy().T
        if norm:
            text_embs /= np.linalg.norm(text_embs, axis=-1, keepdims=True)
        return text_embs
        
    @staticmethod
    def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

class LabelAugment(Augment):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)
        if not self.cfg.AUGMENTATION.GENERIC:
            raise NotImplementedError
        self.dom_components = self.get_domain_component(self.image_features, self.domain_labels, self.cfg.METHOD.NORMALIZE, self.cfg.METHOD.K)

    def augment_single(self, img_embedding, label):
        """
        Augments a single image embedding with the text embedding.
        """
        aug_embedding = np.array([img_embedding] +[img_embedding + feat for feat in self.dom_components])
        aug_embedding /= np.linalg.norm(aug_embedding, axis=-1, keepdims=True)
        return list(aug_embedding)

    @staticmethod
    def get_domain_component(image_features, domain_labels, normalize=True, num_components=1):
        """
        Gets the principle component of the images with a domain.

        args:
            image_features: a numpy list of image embeddings
            domain_labels: a numpy list of domain labels for eahc of the images
            do_pca: if true, do PCA on the image embeddings, else return the mean of the embeddings
        returns:
            domain_components: numpy array of size (# domains, embedding size)
        """
        assert len(image_features) == len(domain_labels), "image_features and domain_labels must be the same length"
        if normalize:
            image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
        groups = {d: [] for d in sorted(np.unique(domain_labels))}
        for emb, label in zip(image_features, domain_labels):
            groups[label].append(emb)
        dom_components = []
        for g in groups:
            if num_components > 0:
                pca = PCA(n_components=num_components)
                pca.fit(np.array(groups[g]))
                dom_components.append(pca.components_[0])
            else:
                dom_components.append(np.mean(np.array(groups[g]), axis=0)[0])
        print("domain components shape: ", np.array(dom_components).shape)
        return np.array(dom_components)

class DiffusionAddition(Addition):
    """
    Does the same thing as Addition but runs the text embeddings through diffusion model.
    """

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)
        if not self.cfg.AUGMENTATION.GENERIC:
            raise NotImplementedError
        text_to_image = torch.load(self.cfg.METHOD.TEXT_TO_IMAGE_PATH)
        text_to_image.requires_grad_(False).eval().to('cpu')
        self.text_features = text_to_image(torch.Tensor(text_features).float()).half().detach().numpy()
        print("text features shape: ", self.text_features.shape)
 
# TO-DO: Implement learned augmentation using CLIP_{direction} loss function  

class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x): 
        # print(x.dtype)
        x = nnf.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Directional(Augment): 
    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features, val_image_features, val_labels, val_group_labels,val_domain_labels, val_filenames):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)
        dataset = EmbeddingDataset(self.cfg, self.image_features, self.labels, self.group_labels, self.domain_labels)
        self.dataset = dataset
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)

        val_dataset = EmbeddingDataset(self.cfg, val_image_features, val_labels, val_group_labels, val_domain_labels)
        self.val_dataset = val_dataset
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets = []
        self.net_checkpoints = []
        self.uid = uuid.uuid4()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.cfg.EXP.IMAGE_FEATURES == 'clip':
            model, preprocess = clip.load(self.cfg.EXP.CLIP_MODEL, device)
        elif self.cfg.EXP.IMAGE_FEATURES == 'openclip':
            model, _, preprocess = open_clip.create_model_and_transforms(self.cfg.EXP.CLIP_MODEL, pretrained=self.cfg.EXP.CLIP_PRETRAINED_DATASET)
            model = model.to(torch.device('cuda:1'))
        # model, preprocess = clip.load(self.cfg.EXP.CLIP_MODEL, device)
        model.eval()
        self.model = model
        if self.cfg.DATA.DATASET == 'ColoredMNISTBinary':
            text_embs = zeroshot_classifier([[f'a photo of the number "{c}"'] for c in self.class_names], model, model_type=self.cfg.EXP.IMAGE_FEATURES)
        else:
            text_embs = zeroshot_classifier([[f"a photo of a {c}"] for c in self.class_names], model, model_type=self.cfg.EXP.IMAGE_FEATURES)
        
        self.class_text_embs = text_embs.float().cuda()
        print("text emb shape ", self.class_text_embs.shape)

        try:
            self.orig_prompts = torch.Tensor(self.get_orig_text_embeddings(self.prompts).transpose((1, 0, 2))).float().cuda()
            self.neutral_embs = torch.Tensor(self.get_orig_text_embeddings(self.neutral_prompts).transpose((1, 0, 2))).float().cuda()
            # if len(self.orig_prompts) > 1:
            #     raise ValueError("this only works for one domain shift atm")
            print('=========================')
            print('=========================')
            print("task specific text emb ", self.orig_prompts.shape, self.orig_prompts[0].shape, torch.transpose(self.orig_prompts[0].float().cuda(), 1, 0).shape, self.class_text_embs.shape)
            print('=========================')
            print('=========================')
            # stack = [torch.mean(self.neutral_embs, dim=1)] + torch.mean(self.orig_prompts, dim=1)
            self.val_dom_check = torch.squeeze(torch.cat([torch.mean(self.neutral_embs, dim=1), torch.mean(self.orig_prompts, dim=1)]), dim=1).float().cuda()
            self.val_dom_check = torch.transpose(self.val_dom_check, 0, 1)
            print("val domain check shape ", self.val_dom_check.shape, self.class_text_embs.shape)
        except:
            print("can't load prompts")

        if not self.cfg.AUGMENTATION.GENERIC:

            if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
                print("DOMAIN SPECIFIC PROMPTS")
                print("task specific text emb ", self.orig_prompts.shape)
                for i in range(len(self.orig_prompts)):
                    self.train_network("sketch", self.orig_prompts[:,i], i)
            else:
                for i in range(len(self.text_features[0])):
                    self.train_network("sketch", self.text_features[:,i], i)
        else:
            for i in range(len(self.text_features)):
                self.train_network("sketch", self.text_features[i], i)

        for net in self.nets: 
            net.eval()
        
    def directional_loss_builder(self, num_net):
        """
        CLIP directional loss from gan NADA paper. Ensures that the difference in
        image embeddings is similar to the difference in text embeddings of the 
        source and target domain.
        """
        if not self.cfg.AUGMENTATION.GENERIC:
            delta_t = torch.Tensor(self.text_features[:,num_net])
        else:
            delta_t = torch.Tensor(self.text_features[num_net])
        delta_t = delta_t.type(torch.float).cuda()
        
        def custom_loss(predictions, labels, targets):
            total_sum = None
            delta_i = predictions - labels
            ctr = 0
            for i, l in zip(delta_i, targets): 
                if not self.cfg.AUGMENTATION.GENERIC:
                    delta_tt = delta_t[l]
                else:
                    delta_tt = delta_t
                ctr += 1
                if total_sum == None: 
                    numerator = torch.dot(i, delta_tt)
                    denominator = torch.norm(i) * torch.norm(delta_tt)
                    total_sum = 1 - (numerator/denominator)
                else: 
                    total_sum += 1 - (torch.dot(i, delta_tt)/ (torch.norm(i) * torch.norm(delta_tt)))
            return total_sum / ctr
        return custom_loss

    @staticmethod
    def get_class_logits(outputs, class_embs):
        outputs_norm = outputs / outputs.norm(dim=-1, keepdim=True) 
        return torch.matmul(outputs_norm, class_embs)

    def train_network(self, source, target, num_net): 
        net = MLP(hidden_dim=self.cfg.AUGMENTATION.MODEL.HIDDEN_DIM, input_dim=self.dataset.embedding_dim)
        self.nets.append(net.cuda())
        self.net_checkpoints.append("")

        cudnn.benchmark = True
        self.optimizer = AdamW(self.nets[num_net].parameters(), lr=self.cfg.AUGMENTATION.MODEL.LR, weight_decay=self.cfg.AUGMENTATION.MODEL.WEIGHT_DECAY)
        self.directional_loss = self.directional_loss_builder(num_net)
        self.class_consistency_loss = nn.CrossEntropyLoss(weight=self.dataset.class_weights.cuda())

        if self.cfg.AUGMENTATION.CLIP_NN_LOSS:
            self.clip_nn_loss = nn.CrossEntropyLoss()

        self.nets[num_net].train()
        
        best_train_loss, best_epoch = 10000, 0
        for epoch in range(self.cfg.AUGMENTATION.EPOCHS):
            train_metrics = self.training_loop(num_net, epoch)
            val_metrics = self.eval_loop(num_net, epoch)
            if val_metrics['val loss'] < best_train_loss:
                    best_train_loss = val_metrics['val loss']
                    best_epoch = epoch
                    self.net_checkpoints[num_net] = self.save_checkpoint(best_train_loss, epoch, num_net)

        wandb.summary[f"{self.prompts[num_net]} best epoch"] = best_epoch
        wandb.summary[f"{self.prompts[num_net]} best train_loss"] = best_train_loss
        print(f"==> loading checkpoint {self.net_checkpoints[num_net]} at epoch {best_epoch} with loss {best_train_loss}")
        self.nets[num_net] = self.load_checkpoint(self.nets[num_net], self.net_checkpoints[num_net])

    def get_nn(self, inputs_unnorm, samples_unnorm, labels, cs=False):
        """ Gets nearest neighbor of that same class for each img_emb"""
        inputs = inputs_unnorm / inputs_unnorm.norm(dim=-1, keepdim=True)
        samples = samples_unnorm / samples_unnorm.norm(dim=-1, keepdim=True)
        nns, nn_dot_prod = [], []
        for i, input in enumerate(inputs):
            if self.cfg.AUGMENTATION.NN_INCLUDE_SAMPLE:
                assert self.cfg.AUGMENTATION.COMPARE_BEFORE_AUG, "Must compare before augmentation"
                nn_features = samples
            else:
                nn_features = torch.cat([samples[0:i], samples[i+1:]])
            dot_prod = input @ nn_features.T
            dot_prod = dot_prod * np.exp(0.007)
            nns.append(torch.argmax(dot_prod))
            nn_dot_prod.append(dot_prod)
        return torch.stack(nns).long(), torch.stack(nn_dot_prod)

    def training_loop(self, num_net, epoch):
        self.nets[num_net].train()
        train_directional_loss, train_class_loss, train_nn_loss, train_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
            self.optimizer.zero_grad()
            cls_outputs = self.nets[num_net](inp)
            # compute directional loss
            directional_loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.directional_loss(cls_outputs, inp, cls_target)

            if not self.cfg.AUGMENTATION.GENERIC:
                if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
                    cls_logits = self.get_class_logits(cls_outputs, torch.transpose(self.orig_prompts[num_net].float().cuda(), 1, 0))
                else:
                    cls_logits = self.get_class_logits(cls_outputs, self.class_text_embs)
                cls_consist = self.class_consistency_loss(cls_logits, cls_target)
                loss = self.alpha * directional_loss + (1 - self.alpha) * cls_consist
                train_class_loss += (1 - self.alpha) * cls_consist
                train_directional_loss += self.alpha * directional_loss.item()
            else:
                train_directional_loss += directional_loss.item()
                # wandb.log({"directional loss": directional_loss.item()})
                loss = directional_loss

            if self.cfg.AUGMENTATION.CLIP_NN_LOSS:
                nn_labels, _ = self.get_nn(inp, inp, cls_target)
                if self.cfg.AUGMENTATION.COMPARE_BEFORE_AUG:
                    _, nn_logits = self.get_nn(cls_outputs, inp, cls_target)
                else:
                    _, nn_logits = self.get_nn(cls_outputs, cls_outputs, cls_target)
                nn_loss = self.cfg.AUGMENTATION.NN_WEIGHT * self.clip_nn_loss(nn_logits, nn_labels)
                loss += nn_loss
                train_nn_loss += nn_loss.item()
        
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            train_loss += loss.item() 

            total += cls_target.size(0)
            progress_bar(i, len(self.train_loader), 'Loss: %.3f'
                            % (train_loss/(i+1)))

        metrics = {"train class loss": train_class_loss/(i+1), "train directional loss": train_directional_loss/(i+1), "train nn loss": train_nn_loss/(i+1), "train loss": train_loss/(i+1), "epoch": epoch}
        wandb.log(metrics)
        return metrics

    def eval_loop(self, num_net, epoch):
        """
        Checkpoint aug netowrk on the eval set. Try both minizing the loss and maximizing the zeroshot accuracy with the correct class and domain.
        """
        m = nn.Softmax(dim=1)
        self.nets[num_net].eval()
        nn_correct, dom_correct, total = 0, 0, 0
        train_directional_loss, train_class_loss, train_nn_loss, train_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.val_loader):
            with torch.no_grad():
                inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
                cls_outputs = self.nets[num_net](inp)
                # compute directional loss
                # directional_loss = self.directional_loss(cls_outputs, inp, cls_target)
                try:
                    cls_logits = self.get_class_logits(cls_outputs, self.val_dom_check)
                    _, cls_predicted = m(cls_logits).max(1)
                    targets = torch.Tensor([1 for _ in range(cls_target.size(0))]).cuda().float()
                    dom_correct += (cls_predicted == targets).sum().item()
                except:
                    dom_correct = 0

                nn_labels, _ = self.get_nn(inp, inp, cls_target)
                nns, nn_logits = self.get_nn(cls_outputs, inp, cls_target)
                nn_correct += nns.eq(nn_labels).sum().item()

                cls_outputs = self.nets[num_net](inp)
                # compute directional loss
                directional_loss = self.directional_loss(cls_outputs, inp, cls_target)
                cls_logits = self.get_class_logits(cls_outputs, self.class_text_embs)
                cls_consist = self.class_consistency_loss(cls_logits, cls_target)
                loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.alpha * directional_loss + (1 - self.alpha) * cls_consist
                train_directional_loss += self.cfg.AUGMENTATION.DOM_WEIGHT * self.alpha * directional_loss.item()

                if self.cfg.AUGMENTATION.CLIP_NN_LOSS:
                    nn_labels, _ = self.get_nn(inp, inp, cls_target)
                    if self.cfg.AUGMENTATION.COMPARE_BEFORE_AUG:
                        _, nn_logits = self.get_nn(cls_outputs, inp, cls_target)
                    else:
                        _, nn_logits = self.get_nn(cls_outputs, cls_outputs, cls_target)
                    nn_loss = self.cfg.AUGMENTATION.NN_WEIGHT * self.clip_nn_loss(nn_logits, nn_labels)
                    train_nn_loss += nn_loss.item()
        
            train_loss += loss.item() 

            total += cls_target.size(0)


        metrics = {"val loss": train_loss/(i+1), "val dom acc": dom_correct/total, "val nn acc": nn_correct/total, "epoch": epoch}
        wandb.log(metrics)
        return metrics

    def eval_lang_loop(self, num_net, epoch):
        self.nets[num_net].eval()
        train_directional_loss, train_class_loss, train_nn_loss, train_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.val_loader):
            inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
        return

    def augment_single(self, img_embedding, label): 
        keep = img_embedding
        if self.cfg.AUGMENTATION.INCLUDE_ORIG_TRAINING:
            output = [keep]
        else:
            output = []
        img_embedding = torch.tensor(img_embedding)
        img_embedding = img_embedding.type(torch.float32)
        img_embedding = img_embedding.cuda()
        img_embedding /= img_embedding.norm(dim=-1, keepdim=True) 
        for net in self.nets: 
            o = net(img_embedding)
            o /= o.norm(dim=-1, keepdim=True) 
            
            o = o.detach().cpu().numpy()
            output.append(o)
            wandb.log({"cos sim:": distance.cosine(o, img_embedding.cpu())})
        # output = self.net(img_embedding)
        # val = torch.tensor(output)
        return list(np.array(output))

    def save_checkpoint(self, acc, epoch, num_net):
        path = f'./checkpoint/{self.prompts[num_net]}-{self.cfg.EXP.SEED}-{self.uid}.pth'
        print(f'Saving checkpoint with acc {acc} to {path}...')
        state = {
            "acc": acc,
            "epoch": epoch,
            "net": self.nets[num_net].state_dict()
        }
        torch.save(state, path)
        wandb.save(path)
        return path

    def load_checkpoint(self, net, path):
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['net'])
        print(f"...loaded checkpoint with acc {checkpoint['acc']}")
        return net

class BiasDirectional(Directional):
    """
    This implements the similar directional loss as the directional class, but routes examples
    based on clips classification of their domain. 
    """

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features, val_image_features, val_labels, val_group_labels,val_domain_labels, val_filenames):
        super(Directional, self).__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features)
        if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
            self.orig_prompts = torch.Tensor(self.get_orig_text_embeddings(self.prompts).transpose((1, 0, 2))).float().cuda()
            print("orig prompts shape ", self.orig_prompts.shape)
            self.text_features = torch.mean(self.orig_prompts, dim=1)
            text_features = self.text_features
        else:
            self.text_features = torch.tensor(text_features).float().cuda()
        print("text features shape ", self.text_features.shape)
        
        print("text features", text_features.shape)
        dataset = EmbeddingDataset(self.cfg, self.image_features, self.labels, self.group_labels, self.domain_labels, text_emb=self.text_features)
        self.dataset = dataset
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets = []
        self.net_checkpoints = []
        self.domain_indexes = [0]
        self.uid = uuid.uuid4()

        if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
            self.class_text_embs = self.orig_prompts
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.cfg.EXP.IMAGE_FEATURES == 'clip':
                model, preprocess = clip.load(self.cfg.EXP.CLIP_MODEL, device)
            elif self.cfg.EXP.IMAGE_FEATURES == 'openclip':
                model, _, preprocess = open_clip.create_model_and_transforms(self.cfg.EXP.CLIP_MODEL, pretrained=self.cfg.EXP.CLIP_PRETRAINED_DATASET)
                model = model.to(torch.device('cuda:1'))
            # model, preprocess = clip.load(self.cfg.EXP.CLIP_MODEL, device)
            model.eval()
            self.model = model
            text_embs = zeroshot_classifier([[f"a photo of a {c}"] for c in self.class_names], model, model_type=self.cfg.EXP.IMAGE_FEATURES, cuda_device='1')
            self.class_text_embs = text_embs.float().cuda()

        print("text emb shape ", self.class_text_embs.shape)

        print("text features ", self.text_features.shape)
        self.train_network("sketch", self.text_features, 0)

        for net in self.nets: 
            net.eval()
        
    def directional_loss_builder(self, num_net):
        """
        CLIP directional loss from gan NADA paper. Ensures that the difference in
        image embeddings is similar to the difference in text embeddings of the 
        source and target domain.
        This modification changes the delta depending on
        """
        def custom_loss(predictions, labels, targets, domain_labels):
            total_sum = None
            delta_i = predictions - labels
            ctr = 0
            for i, d, l in zip(delta_i, domain_labels, targets): 
                if d == 0:
                    delta_tt = self.text_features[1] - self.text_features[0]
                    # delta_tt = self.neutral_embedding[0][l] - self.orig_prompts[0][l]
                else:
                    delta_tt = self.text_features[0] - self.text_features[1]
                    # delta_tt = self.neutral_embedding[0][l] - self.orig_prompts[1][l]
                try:
                    delta_tt /= np.linalg.norm(delta_tt, axis=-1, keepdims=True)
                    delta_tt = torch.Tensor(delta_tt).type(torch.float).cuda()
                except:
                    delta_tt /= delta_tt.norm(dim=-1, keepdim=True)
                ctr += 1
                if total_sum == None: 
                    numerator = torch.dot(i, delta_tt)
                    denominator = torch.norm(i) * torch.norm(delta_tt)
                    total_sum = 1 - (numerator/denominator)
                else: 
                    total_sum += 1 - (torch.dot(i, delta_tt)/ (torch.norm(i) * torch.norm(delta_tt)))
            return total_sum / ctr
        return custom_loss

    def loss_builder(self): 
        cos = torch.nn.CosineSimilarity(dim=0)
        def custom_loss(predictions, inputs, labels): 
            total_sum = None
            ctr = 0
            for p, _, l in zip(predictions, inputs, labels): 
                # get class similarity
                dom_one_sim = cos(p, torch.Tensor(self.text_features[0]).cuda())

                # get sketch similarity
                dom_two_sim = cos(p, torch.Tensor(self.text_features[1]).cuda())

                loss = torch.abs(dom_one_sim - dom_two_sim)
                if total_sum == None: 
                    total_sum = loss
                else: 
                    total_sum += loss
                ctr += 1
            return total_sum / ctr
        return custom_loss

    @staticmethod
    def get_class_logits(outputs, class_embs, dom_labels, dom_specific=False):
        outputs_norm = outputs / outputs.norm(dim=-1, keepdim=True) 
        if dom_specific:
            ret = []
            for o, d in zip(outputs, dom_labels):
                idx = 1 if d == 0 else 0
                ret.append(torch.matmul(o, class_embs[idx].transpose(0, 1)))
            return torch.stack(ret).cuda()
        else:
            return torch.matmul(outputs_norm, class_embs)

    def train_network(self, source, target, num_net): 
        net = MLP(hidden_dim=self.cfg.AUGMENTATION.MODEL.HIDDEN_DIM, input_dim=self.dataset.embedding_dim)
        self.nets.append(net.cuda())
        self.net_checkpoints.append("")

        cudnn.benchmark = True
        self.optimizer = AdamW(self.nets[num_net].parameters(), lr=self.cfg.AUGMENTATION.MODEL.LR, weight_decay=self.cfg.AUGMENTATION.MODEL.WEIGHT_DECAY)
        if self.cfg.AUGMENTATION.CS_LOSS:
            self.directional_loss = self.loss_builder()
        else:
            self.directional_loss = self.directional_loss_builder(num_net)
        self.class_consistency_loss = nn.CrossEntropyLoss(weight=self.dataset.class_weights.cuda())

        if self.cfg.AUGMENTATION.CLIP_NN_LOSS:
            self.clip_nn_loss = nn.CrossEntropyLoss()

        self.nets[num_net].train()
        
        best_train_loss, best_epoch = 10000, 0
        for epoch in range(self.cfg.AUGMENTATION.EPOCHS):
            train_metrics = self.training_loop(num_net, epoch)
            if train_metrics['train loss'] < best_train_loss:
                    best_train_loss = train_metrics['train loss']
                    best_epoch = epoch
                    self.net_checkpoints[num_net] = self.save_checkpoint(best_train_loss, epoch, num_net)

        wandb.summary[f"{self.prompts[num_net]} best epoch"] = best_epoch
        wandb.summary[f"{self.prompts[num_net]} best train_loss"] = best_train_loss

    def training_loop(self, num_net, epoch):
        train_directional_loss, train_class_loss, train_nn_loss, train_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
            self.optimizer.zero_grad()
            cls_outputs = self.nets[num_net](inp)
            # compute directional loss
            if self.cfg.AUGMENTATION.CS_LOSS:
                directional_loss = self.directional_loss(cls_outputs, inp, cls_target)
            else:
                directional_loss = self.directional_loss(cls_outputs, inp, cls_target, dom_target)

            cls_logits = self.get_class_logits(cls_outputs, self.class_text_embs, dom_target, self.cfg.AUGMENTATION.DOM_SPECIFIC_XE)
            cls_consist = self.class_consistency_loss(cls_logits, cls_target)
            loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.alpha * directional_loss + (1 - self.alpha) * cls_consist
            train_class_loss += (1 - self.alpha) * cls_consist
            train_directional_loss += self.cfg.AUGMENTATION.DOM_WEIGHT * self.alpha * directional_loss.item()

            if self.cfg.AUGMENTATION.CLIP_NN_LOSS:
                nn_labels, _ = self.get_nn(inp, inp, cls_target)
                if self.cfg.AUGMENTATION.COMPARE_BEFORE_AUG:
                    _, nn_logits = self.get_nn(cls_outputs, inp, cls_target)
                else:
                    _, nn_logits = self.get_nn(cls_outputs, cls_outputs, cls_target)
                nn_loss = self.cfg.AUGMENTATION.NN_WEIGHT * self.clip_nn_loss(nn_logits, nn_labels)
                loss += nn_loss
                train_nn_loss += nn_loss.item()
        
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            train_loss += loss.item() 

            total += cls_target.size(0)
            progress_bar(i, len(self.train_loader), 'Loss: %.3f'
                            % (train_loss/(i+1)))

        metrics = {"train class loss": train_class_loss/(i+1), "train directional loss": train_directional_loss/(i+1), "train nn loss": train_nn_loss/(i+1), "train loss": train_loss/(i+1), "epoch": epoch}
        wandb.log(metrics)
        return metrics

    @staticmethod
    def get_inv(label):
        return 1 if label == 0 else 0

    def augment_dataset(self):
            """
            Augments the dataset
            """
            augmented_features = []
            augmented_labels = []
            augmented_domain_labels = []
            augmented_group_labels = []
            augmented_filenames = []
            for i, feature in enumerate(self.image_features):
                augmented_features += self.augment_single(feature, self.labels[i])
                augmented_labels += [self.labels[i], self.labels[i]]
                augmented_domain_labels += [self.domain_labels[i], self.get_inv(self.domain_labels[i])]
                augmented_group_labels += [self.group_labels[i], self.get_inv(self.group_labels[i])]
                augmented_filenames += [self.filenames[i], self.filenames[i]]
            return np.array(augmented_features), np.array(augmented_labels), np.array(augmented_domain_labels), np.array(augmented_group_labels), np.array(augmented_filenames)

class ProbeLoss:
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def __call__(self, encodings, labels):
        # print("eoncdings shape ", encodings.shape)
        inp = encodings.detach() / encodings.detach().norm(dim=-1, keepdim=True) 
        outputs = self.net(inp.cuda())
        return self.criterion(outputs, labels)

class ProbeDirectional(Directional):

    def __init__(self, cfg, image_features, labels, group_labels, domain_labels, filenames, text_features, val_image_features, val_labels, val_group_labels,val_domain_labels, val_filenames):
        super().__init__(cfg, image_features, labels, group_labels, domain_labels, filenames, text_features, val_image_features, val_labels, val_group_labels,val_domain_labels, val_filenames)

    def train_network(self, source, target, num_net): 
        net = MLP(hidden_dim=self.cfg.AUGMENTATION.MODEL.HIDDEN_DIM, input_dim=self.dataset.embedding_dim)
        self.nets.append(net.cuda())
        self.net_checkpoints.append("")

        self.optimizer = AdamW(self.nets[num_net].parameters(), lr=self.cfg.AUGMENTATION.MODEL.LR, weight_decay=self.cfg.AUGMENTATION.MODEL.WEIGHT_DECAY)
        self.directional_loss = self.directional_loss_builder(num_net)

        print("LOADING CLASSIFIER......")
        B, W  = self.image_features.shape
        print("shape ", B, W)
        model_conf = OmegaConf.create({"in_dim": W, "h_dim": W, "out_dim": self.dataset.num_classes, "num_classes": self.dataset.num_classes, "num_domains": self.dataset.num_domains, "num_layers": self.cfg.METHOD.MODEL.NUM_LAYERS})
        cfg = OmegaConf.merge(self.cfg, model_conf)
        self.classifier = self.load_checkpoint(predictors.MLP(cfg), self.cfg.AUGMENTATION.CLASSIFIER_CHECKPOINT).cuda()
        self.classifier.eval()
        print("DONE")

        self.class_consistency_loss = ProbeLoss(self.classifier, nn.CrossEntropyLoss(weight=self.dataset.class_weights.cuda()))

        self.nets[num_net].train()
        
        best_train_loss, best_epoch = 10000, 0
        for epoch in range(self.cfg.AUGMENTATION.EPOCHS):
            train_metrics = self.training_loop(num_net, epoch)
            val_metrics = self.eval_loop(num_net, epoch)
            if val_metrics['val loss'] < best_train_loss:
                    best_train_loss = val_metrics['val loss']
                    best_epoch = epoch
                    self.net_checkpoints[num_net] = self.save_checkpoint(best_train_loss, epoch, num_net)

        wandb.summary[f"{self.prompts[num_net]} best epoch"] = best_epoch
        wandb.summary[f"{self.prompts[num_net]} best train_loss"] = best_train_loss
        print(f"==> loading checkpoint {self.net_checkpoints[num_net]} at epoch {best_epoch} with loss {best_train_loss}")
        self.nets[num_net] = self.load_checkpoint(self.nets[num_net], self.net_checkpoints[num_net])

    def training_loop(self, num_net, epoch):
        self.nets[num_net].train()
        train_directional_loss, train_class_loss, train_nn_loss, train_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.train_loader):
            inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
            self.optimizer.zero_grad()
            cls_outputs = self.nets[num_net](inp)
            # cls_outputs /= cls_outputs.norm(dim=-1, keepdim=True) 

            # compute directional loss
            directional_loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.directional_loss(cls_outputs, inp, cls_target)

            if not self.cfg.AUGMENTATION.GENERIC:
                if self.cfg.AUGMENTATION.DOM_SPECIFIC_XE:
                    cls_logits = self.get_class_logits(cls_outputs, torch.transpose(self.orig_prompts[num_net].float().cuda(), 1, 0))
                else:
                    cls_logits = self.get_class_logits(cls_outputs, self.class_text_embs)
                # cls_consist = self.class_consistency_loss(cls_logits, cls_target)
                cls_consist = self.class_consistency_loss(cls_outputs, cls_target)
                loss = self.alpha * directional_loss + (1 - self.alpha) * cls_consist
                train_class_loss += (1 - self.alpha) * cls_consist
                train_directional_loss += self.alpha * directional_loss.item()
            else:
                train_directional_loss += directional_loss.item()
                # wandb.log({"directional loss": directional_loss.item()})
                cls_consist = self.class_consistency_loss(cls_outputs, cls_target)
                loss = self.alpha * directional_loss + (1 - self.alpha) * cls_consist

            if self.cfg.AUGMENTATION.CLIP_NN_LOSS:
                nn_labels, _ = self.get_nn(inp, inp, cls_target)
                if self.cfg.AUGMENTATION.COMPARE_BEFORE_AUG:
                    _, nn_logits = self.get_nn(cls_outputs, inp, cls_target)
                else:
                    _, nn_logits = self.get_nn(cls_outputs, cls_outputs, cls_target)
                nn_loss = self.cfg.AUGMENTATION.NN_WEIGHT * self.clip_nn_loss(nn_logits, nn_labels)
                loss += nn_loss
                train_nn_loss += nn_loss.item()
        
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            train_loss += loss.item() 

            total += cls_target.size(0)
            progress_bar(i, len(self.train_loader), 'Loss: %.3f'
                            % (train_loss/(i+1)))

        metrics = {"train class loss": train_class_loss/(i+1), "train directional loss": train_directional_loss/(i+1), "train nn loss": train_nn_loss/(i+1), "train loss": train_loss/(i+1), "epoch": epoch}
        wandb.log(metrics)
        return metrics

    def eval_loop(self, num_net, epoch):
        """
        Checkpoint aug netowrk on the eval set. Try both minizing the loss and maximizing the zeroshot accuracy with the correct class and domain.
        """
        m = nn.Softmax(dim=1)
        self.nets[num_net].eval()
        nn_correct, dom_correct, total = 0, 0, 0
        train_directional_loss, train_class_loss, train_nn_loss, train_loss, cls_correct, total = 0, 0, 0, 0, 0, 0
        for i, (inp, cls_target, cls_group, dom_target) in enumerate(self.val_loader):
            with torch.no_grad():
                inp, cls_target= inp.cuda().float(), cls_target.cuda().long()
                cls_outputs = self.nets[num_net](inp)
                cls_outputs /= cls_outputs.norm(dim=-1, keepdim=True) 
                # compute directional loss
                # directional_loss = self.directional_loss(cls_outputs, inp, cls_target)
                try:
                    cls_logits = self.get_class_logits(cls_outputs, self.val_dom_check)
                    _, cls_predicted = m(cls_logits).max(1)
                    targets = torch.Tensor([1 for _ in range(cls_target.size(0))]).cuda().float()
                    dom_correct += (cls_predicted == targets).sum().item()
                except:
                    dom_correct = 0

                nn_labels, _ = self.get_nn(inp, inp, cls_target)
                nns, nn_logits = self.get_nn(cls_outputs, inp, cls_target)
                nn_correct += nns.eq(nn_labels).sum().item()

                cls_outputs = self.nets[num_net](inp)
                # compute directional loss
                directional_loss = self.directional_loss(cls_outputs, inp, cls_target)
                cls_logits = self.get_class_logits(cls_outputs, self.class_text_embs)
                cls_consist = self.class_consistency_loss(cls_outputs, cls_target)
                loss = self.cfg.AUGMENTATION.DOM_WEIGHT * self.alpha * directional_loss + (1 - self.alpha) * cls_consist
                train_directional_loss += self.cfg.AUGMENTATION.DOM_WEIGHT * self.alpha * directional_loss.item()

                if self.cfg.AUGMENTATION.CLIP_NN_LOSS:
                    nn_labels, _ = self.get_nn(inp, inp, cls_target)
                    if self.cfg.AUGMENTATION.COMPARE_BEFORE_AUG:
                        _, nn_logits = self.get_nn(cls_outputs, inp, cls_target)
                    else:
                        _, nn_logits = self.get_nn(cls_outputs, cls_outputs, cls_target)
                    nn_loss = self.cfg.AUGMENTATION.NN_WEIGHT * self.clip_nn_loss(nn_logits, nn_labels)
                    train_nn_loss += nn_loss.item()
        
            train_loss += loss.item() 

            total += cls_target.size(0)


        metrics = {"val loss": train_loss/(i+1), "val dom acc": dom_correct/total, "val nn acc": nn_correct/total, "epoch": epoch}
        wandb.log(metrics)
        return metrics