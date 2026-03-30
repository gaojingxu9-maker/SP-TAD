from itertools import groupby
from operator import itemgetter
import numpy as np
from metrics.metrics import *
from metrics.combine_all_scores import combine_all_evaluation_scores
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from utils.affiliation import convert_vector_to_events
from utils.affiliation import pr_from_events

def getAffiliationMetrics(label, pred):
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(label)
    Trange = (0, len(pred))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    if P + R > 0.0:
        F = 2 * P * R / (P + R)
    else:
        F = 0.0
    return P, R, F
def point_adujest(test_labels, test_energy):
    Accuracy = 0
    F1 = 0
    Precision = 0
    Recall = 0
    thr = 0
    thre = [(90 + (i / 10)) for i in range(100)]
    thres = np.percentile(test_energy, thre)
    for i in thres:
        thresh = i
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        pred = np.array(pred)
        gt = np.array(gt)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                             average='binary')




        if f_score > F1:
            Accuracy = accuracy
            F1 = f_score
            Precision = precision
            Recall = recall
            thr = thresh
    return Accuracy,Precision, Recall,F1,thr
def PAK_AUC(gt, pred):
    f1_pa_k = []
    prec_pa_k = []
    recall_pa_k = []
    gt_idx = np.where(gt == 1)[0]
    for k in range(0, 101):
        anomalies = []
        new_preds = np.array(pred)
        # Find the continuous index
        for _, g in groupby(enumerate(gt_idx), lambda x: x[0] - x[1]):
            anomalies.append(list(map(itemgetter(1), g)))
        # For each anomaly (point or seq) in the test dataset
        for a in anomalies:
            # Find the predictions where the anomaly falls
            pred_labels = new_preds[a]
            # Check how many anomalies have been predicted (ratio)
            if len(np.where(pred_labels == 1)[0]) / len(a) > (k / 100):
                # Update the whole prediction range as correctly predicted
                new_preds[a] = 1
        new_preds = np.array(new_preds)
        gt = np.array(gt)



        prec_pa, recall_pa, f1_pa, support = precision_recall_fscore_support(gt,new_preds, average='binary')



        f1_pa_k.append(f1_pa)
        prec_pa_k.append(prec_pa)
        recall_pa_k.append(recall_pa)
    f1_pak_auc = np.trapz(f1_pa_k)
    f1_pak_auc = f1_pak_auc / 100
    prec_pak_auc = np.trapz(prec_pa_k)
    prec_pak_auc = prec_pak_auc / 100
    recall_pak_auc = np.trapz(recall_pa_k)
    recall_pak_auc = recall_pak_auc / 100
    return prec_pak_auc, recall_pak_auc, f1_pak_auc


def evaluation_metrics(test_labels, test_energy,test,dataset,id=0,ids=0):
    '''
    test_labels:真实标签
    test_energy：异常得分
    '''
    precision, recall, thresholds = precision_recall_curve(test_labels, test_energy)
    auc_roc = roc_auc_score(test_labels, test_energy)
    auc_pr = auc(recall, precision)

    pak_acc,pak_pre, pak_rec,pak_f_score,thresh= point_adujest(test_labels, test_energy)
    gt = test_labels.astype(int)
    pred = (test_energy > thresh).astype(int)
    prec_pak_auc, recall_pak_auc, pak_f1_auc = PAK_AUC(gt, pred)

    aff_pre, aff_rec, aff_f1 = getAffiliationMetrics(test_labels.copy(), pred.copy())

    scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)



    print("AUC_ROC : {:0.4f}, AUC_PR : {:0.4f}, Aff_Pre : {:0.4f}, Aff_Rec : {:0.4f}, Aff_F1 : {:0.4f} ".format(auc_roc, auc_pr, aff_pre, aff_rec,aff_f1))



    print("R_AUC_ROC : {:0.4f}, R_AUC_PR : {:0.4f}, VUS_ROC : {:0.4f}, VUS_PR : {:0.4f} ".format(scores_simple["R_AUC_ROC"], scores_simple["R_AUC_PR"], scores_simple["VUS_ROC"], scores_simple["VUS_PR"]))
    print("AUC_F1 : {:0.4f}, Acc : {:0.4f}, Pre : {:0.4f}, Rec : {:0.4f}, F1-score : {:0.4f} ".format(pak_f1_auc, pak_acc, pak_pre, pak_rec,pak_f_score))

    if test == 1:
        f = open("result.txt", 'a')
        f.write("数据集  " + str(dataset) + "  \n")
        f.write("lambda1" + str(id) + " lambda2" + str(ids)+ "  \n")

        f.write(
            "Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC-ROC : {:0.4f}, AUC-PR : {:0.4f}, ACC : {:0.4f}, Aff-Pre : {:0.4f}, Aff-Rec : {:0.4f}, Aff-F1 : {:0.4f} ".format(
                pak_pre, pak_rec, pak_f_score, auc_roc, auc_pr, pak_acc, aff_pre, aff_rec, aff_f1) + "  \n")
        # 添加R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, F1_AUC这些指标
        f.write(
            "R_AUC_ROC : {:0.4f}, R_AUC_PR : {:0.4f}, VUS_ROC : {:0.4f}, VUS_PR : {:0.4f}, F1_AUC : {:0.4f} ".format(
                scores_simple["R_AUC_ROC"], scores_simple["R_AUC_PR"], scores_simple["VUS_ROC"],
                scores_simple["VUS_PR"], pak_f1_auc) + "  \n")


        f.write('\n')
        f.close()

    return {"auc_roc": auc_roc, "auc_pr":auc_pr,"aff_pre": aff_pre, "aff_rec":aff_rec,
            "aff_f1": aff_f1, "pak_f1_auc":pak_f1_auc, "pak_acc":pak_acc,
            "pak_pre": pak_pre, "pak_rec":pak_rec, "pak_f_score":pak_f_score}
