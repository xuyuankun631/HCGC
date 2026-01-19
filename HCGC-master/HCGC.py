import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import Encoder_Net
import torch.nn.functional as F
import json

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=4, help="Number of gnn layers") # t = 4
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.') # epoch = 400
parser.add_argument('--dims', type=int, default=500, help='feature dim') # default 500
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='bat', help='name of dataset.')
parser.add_argument('--cluster_num', type=int, default=4, help='number of cluster.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
parser.add_argument('--threshold', type=float, default=0.7, help='the threshold of high-confidence')
parser.add_argument('--threshold2', type=float, default=0.98, help='the threshold of high-confidence')
parser.add_argument('--threshold3', type=float, default=0.99, help='the threshold of high-confidence')
parser.add_argument('--alpha', type=float, default=10.0, help='trade-off of loss')
args = parser.parse_args()



#load graph data
features, true_labels, adj = load_graph_data(args.dataset, show_details=False)


# Laplacian Smoothing
adj_norm_s = preprocess_graph_RandomWalk_Weight(adj, args.t, norm='sym', renorm=True)
smooth_fea = sp.csr_matrix(features).toarray()
for a in adj_norm_s:
    smooth_fea = a.dot(smooth_fea)
smooth_fea = torch.FloatTensor(smooth_fea)

acc_list = []
nmi_list = []
ari_list = []
f1_list = []

accbest_list = []
nmibest_list = []
aribest_list = []
f1best_list =  []


for seed in range(10):
    setup_seed(seed)

    # init
    best_acc, best_nmi, best_ari, best_f1, predict_labels, dis= clustering(smooth_fea, true_labels, args.cluster_num)
    predict_labels2, dis1 = hkclustering(smooth_fea, 50)

    # MLP
    model = Encoder_Net(args.linlayers, [features.shape[1]] + [args.dims])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # GPU
    model.to(args.device)
    smooth_fea = smooth_fea.to(args.device)
    sample_size = features.shape[0]
    target = torch.eye(smooth_fea.shape[0]).to(args.device)
    for epoch in tqdm(range(args.epochs)):
        model.train()
        z1, z2 = model(smooth_fea)
        if epoch > 50:

            high_confidence = torch.min(dis1, dim=1).values
            threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]

            # pos samples
            index = torch.tensor(range(smooth_fea.shape[0]), device=args.device)[high_confidence_idx]
            y_sam = torch.tensor(predict_labels2, device=args.device)[high_confidence_idx]
            index = index[torch.argsort(y_sam)]
            class_num = {}

            for label in torch.sort(y_sam).values:
                label = label.item()
                if label in class_num.keys():
                    class_num[label] += 1
                else:
                    class_num[label] = 1
            key = sorted(class_num.keys())
            if len(class_num) < 2:
                continue
            pos_contrastive = 0
            centers_1 = torch.tensor([], device=args.device)
            centers_2 = torch.tensor([], device=args.device)


            for i in range(len(key[:-1])):
                class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
                now = index[class_num[key[i]]:class_num[key[i + 1]]]
                pos_embed_1 = z1[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                pos_embed_2 = z2[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
                pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()
                centers_1 = torch.cat([centers_1, torch.mean(z1[now], dim=0).unsqueeze(0)], dim=0)
                centers_2 = torch.cat([centers_2, torch.mean(z2[now], dim=0).unsqueeze(0)], dim=0)


            c1_1 = centers_1
            c1_2 = centers_2
            hidden_emb_Cluster2 = (centers_1 + centers_2) / 2
            predict_labels_Cluster2, dis_Cluster2 = hkclustering(hidden_emb_Cluster2, 20)
            dis_ClusterNor2 = F.normalize(dis_Cluster2, dim=1, p=2)
            pos_contrastive = pos_contrastive / len(key)
            loss1 = pos_contrastive
            high_confidence2 = torch.min(dis_ClusterNor2, dim=1).values
            threshold2 = torch.sort(high_confidence2).values[int(len(high_confidence2) * args.threshold2)] #args.threshold
            high_confidence_idx2 = np.argwhere(high_confidence2 < threshold2)[0]
            index2 = torch.tensor(range(hidden_emb_Cluster2.shape[0]), device=args.device)[high_confidence_idx2]
            y_sam2 = torch.tensor(predict_labels_Cluster2, device=args.device)[high_confidence_idx2]
            index2 = index2[torch.argsort(y_sam2)]
            class_num2 = {}

            for label in torch.sort(y_sam2).values:
                label = label.item()
                if label in class_num2.keys():
                    class_num2[label] += 1
                else:
                    class_num2[label] = 1
            key2 = sorted(class_num2.keys())
            if len(class_num2) < 2:
                continue
            pos_contrastive2 = 0
            centers2_1 = torch.tensor([], device=args.device)
            centers2_2 = torch.tensor([], device=args.device)
            for i in range(len(key2[:-1])):
                class_num2[key2[i + 1]] = class_num2[key2[i]] + class_num2[key2[i + 1]]
                now2 = index2[class_num2[key2[i]]:class_num2[key2[i + 1]]]
                pos_embed2_1 = c1_1[np.random.choice(now2.cpu(), size=int((now2.shape[0] * 1.0)), replace=False)]
                pos_embed2_2 = c1_2[np.random.choice(now2.cpu(), size=int((now2.shape[0] * 1.0)), replace=False)]
                pos_contrastive2 += (2 - 2 * torch.sum(pos_embed2_1 * pos_embed2_2,dim=1)).sum()
                centers2_1 = torch.cat([centers2_1, torch.mean(c1_1[now2], dim=0).unsqueeze(0)], dim=0)
                centers2_2 = torch.cat([centers2_2, torch.mean(c1_2[now2], dim=0).unsqueeze(0)], dim=0)

            c2_1 = centers2_1
            c2_2 = centers2_2
            hidden_emb_Cluster3 = (centers2_1 + centers2_2) / 2
            predict_labels_Cluster3, dis_Cluster3 = hkclustering(hidden_emb_Cluster3, args.cluster_num)
            dis_ClusterNor3 = F.normalize(dis_Cluster3, dim=1, p=2)

            pos_contrastive2 = pos_contrastive2 / len(key2)
            loss2 = pos_contrastive2
            high_confidence3 = torch.min(dis_ClusterNor3, dim=1).values
            threshold3 = torch.sort(high_confidence3).values[int(len(high_confidence3) * args.threshold3)]
            high_confidence_idx3 = np.argwhere(high_confidence3 < threshold3)[0]

            # pos samples
            index3 = torch.tensor(range(hidden_emb_Cluster3.shape[0]), device=args.device)[high_confidence_idx3]
            y_sam3 = torch.tensor(predict_labels_Cluster3, device=args.device)[high_confidence_idx3]
            index3 = index3[torch.argsort(y_sam3)]
            class_num3 = {}

            for label in torch.sort(y_sam3).values:
                label = label.item()
                if label in class_num3.keys():
                    class_num3[label] += 1
                else:
                    class_num3[label] = 1
            key3 = sorted(class_num3.keys())
            if len(class_num3) < 2:
                continue
            pos_contrastive3 = 0
            centers3_1 = torch.tensor([], device=args.device)
            centers3_2 = torch.tensor([], device=args.device)

            for i in range(len(key3[:-1])):
                class_num3[key3[i + 1]] = class_num3[key3[i]] + class_num3[key3[i + 1]]
                now3 = index3[class_num3[key3[i]]:class_num3[key3[i + 1]]]
                pos_embed3_1 = c2_1[np.random.choice(now3.cpu(), size=int((now3.shape[0] * 1.0)), replace=False)]
                pos_embed3_2 = c2_2[np.random.choice(now3.cpu(), size=int((now3.shape[0] * 1.0)), replace=False)]
                pos_contrastive3 += (2 - 2 * torch.sum(pos_embed3_1 * pos_embed3_2,dim=1)).sum()
                centers3_1 = torch.cat([centers3_1, torch.mean(c2_1[now3], dim=0).unsqueeze(0)], dim=0)
                centers3_2 = torch.cat([centers3_2, torch.mean(c2_2[now3], dim=0).unsqueeze(0)], dim=0)

            pos_contrastive3 = pos_contrastive3 / len(key3)
            if pos_contrastive3 == 0:
                continue
            if len(class_num3) < 2:
                loss = pos_contrastive3
            else:
                centers3_1 = F.normalize(centers3_1, dim=1, p=2)
                centers3_2 = F.normalize(centers3_2, dim=1, p=2)
                S3 = centers3_1 @ centers3_2.T
                S3_diag = torch.diag_embed(torch.diag(S3))
                S3 = S3 - S3_diag
                neg_contrastive3 = F.mse_loss(S3, torch.zeros_like(S3))
                loss3 = pos_contrastive3 + args.alpha * neg_contrastive3

            loss = loss1 + loss2 + loss3

        else:
            S = z1 @ z2.T
            loss = F.mse_loss(S, target)

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            z1, z2 = model(smooth_fea)

            hidden_emb = (z1 + z2) / 2

            predict_labels2, dis1 = hkclustering(hidden_emb, 50)
            acc, nmi, ari, f1, predict_labels, dis = clustering(hidden_emb, true_labels, args.cluster_num)
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1

            print("Exp",seed,"epo",epoch,"acc:{:.4f}".format(acc),"nmi:{:.4f}".format(nmi),"ari:{:.4f}".format(ari),"f1:{:.4f}".format(f1))


    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)


    print("best_acc:", best_acc, "best_nmi:", best_nmi, "best_ari", best_ari, "best_f1:", best_f1)


accbest_list = np.sort(acc_list)
nmibest_list = np.sort(nmi_list)
aribest_list = np.sort(ari_list)
f1best_list = np.sort(f1_list)

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)


file = open("result/{}/test.csv".format(args.dataset), "a+")


print('{}'.format(args.dataset),'acc: {:.4f} ± {:.4f}'.format(acc_list.mean(), acc_list.std()),'Setting: epoch:{}, lr:{}, thres:{}, alpha:{}'.format(args.epochs, args.lr, args.threshold, args.alpha), file=file)
print('{}'.format(args.dataset),'nmi: {:.4f} ± {:.4f}'.format(nmi_list.mean(), nmi_list.std()), file=file)
print('{}'.format(args.dataset),'ari: {:.4f} ± {:.4f}'.format(ari_list.mean(), ari_list.std()), file=file)
print('{}'.format(args.dataset),'f1: {:.4f} ± {:.4f}'.format(f1_list.mean(), f1_list.std()), file=file)
print('{}'.format(args.dataset),'First Best-> acc: {:.4f}'.format(accbest_list[9]), 'nmi: {:.4f}'.format(nmibest_list[9]), 'ari: {:.4f}'.format(aribest_list[9]), 'f1: {:.4f}'.format(f1best_list[9]), file=file)
print('{}'.format(args.dataset),'Second Best-> acc: {:.4f}'.format(accbest_list[8]), 'nmi: {:.4f}'.format(nmibest_list[8]), 'ari: {:.4f}'.format(aribest_list[8]), 'f1: {:.4f}'.format(f1best_list[8]), file=file)
print('{}'.format(args.dataset),'Third Best-> acc: {:.4f}'.format(accbest_list[7]), 'nmi: {:.4f}'.format(nmibest_list[7]), 'ari: {:.4f}'.format(aribest_list[7]), 'f1: {:.4f}'.format(f1best_list[7]), file=file)


file.close()

params = {
    'acc': accbest_list[9],
    'nmi': nmibest_list[9],
    'ari': aribest_list[9],
    'f1': f1best_list[9],
    'datasetname': args.dataset
}
with open('params.json', 'a') as f:
    json.dump(params, f)
f.close()