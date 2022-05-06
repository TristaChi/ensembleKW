import numpy as np
import argparse
import csv


def readFile(count=2,dir="evalData/l_inf/mnist_small_0_1/mnist_small_0_1",data="test"):
    file=dir+"_"+str(count)+"_"+str(data)
    result = np.loadtxt(file)
    y_pred = result[:,1]
    y_true = result[:,2]
    certified = result[:,3]
    return y_pred, y_true,certified

def robust_voting(y_pred, y_true,certified):
    acc = 0
    vra = 0
    for i in range(np.shape(y_pred)[1]):
        count_vra = np.zeros(int(max(y_pred[0]))+1)
        bot = np.shape(certified)[0] - sum([row[i] for row in certified])
        count_vra[int(y_true[0][i])] -= bot # add bottom to count_vra

        count_acc = np.zeros(int(max(y_pred[0]))+1)

        for j in range(np.shape(y_pred)[0]):
            if certified[j][i] == 1: count_vra[int(y_pred[j][i])] += 1
            count_acc[int(y_pred[j][i])] += 1

        # j - j_true
        count_vra -= np.full_like(count_vra,count_vra[int(y_true[0][i])])
        count_vra[int(y_true[0][i])]=-1

        count_acc -= np.full_like(count_acc,count_acc[int(y_true[0][i])])
        count_acc[int(y_true[0][i])]=-1

        # j_true is j_max 
        if sum(count_vra<0)==len(count_vra): vra += 1
        if sum(count_acc<0)==len(count_acc): acc += 1

    print("voting clean_acc: ", acc/np.shape(y_pred)[1],", error: ",1-acc/np.shape(y_pred)[1])
    print("voting vra: ", vra/np.shape(y_pred)[1],", error: ",1-vra/np.shape(y_pred)[1])

def cascade(y_pred, y_true,certified):
    correct = 0
    vra = 0
    for i in range(np.shape(y_pred)[1]):
        for j in range(np.shape(y_pred)[0]):
            if y_pred[j][i] == y_true[j][i] and certified[j][i] == 1:
                correct = correct + 1
                vra = vra +1
                break
            elif y_pred[j][i] == y_true[j][i] and j==np.shape(y_pred)[0]-1:
                correct = correct + 1

    print("cascade clean_acc: ", correct/np.shape(y_pred)[1],", error: ",1-correct/np.shape(y_pred)[1])
    print("cascade vra: ", vra/np.shape(y_pred)[1],", error: ",1-vra/np.shape(y_pred)[1])

    

if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--filename', default=opt)
    # args = parser.parse_args()

    # readFile(args.filename)
    model_type="cifar_small_0_2"
    dir = "/home/chi/NNRobustness/ensembleKW/evalData/l_inf/"+model_type+"/test_"+model_type
    count=3
    y_pred_all = []
    y_true_all = []
    certified_all =[]
    print("results for model type ", model_type)
    for i in range(count):
        y_pred, y_true,certified = readFile(count=i,dir=dir,data="test")
        y_pred_all.append(y_pred)
        y_true_all.append(y_true)
        certified_all.append(certified)
        
    print("model 0 clean accuracy: ", np.mean(y_pred_all[0]==y_true_all[0]),", error: ",1-np.mean(y_pred_all[0]==y_true_all[0]))
    print("model 0 vra: ", np.mean((y_pred_all[0]==y_true_all[0])*certified_all[0]),", error: ",1-np.mean((y_pred_all[0]==y_true_all[0])*certified_all[0]))

    cascade(y_pred_all, y_true_all,certified_all)
    robust_voting(y_pred_all, y_true_all,certified_all)