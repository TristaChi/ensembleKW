import numpy as np
import argparse

def readFile(count=2,dir="/longterm/chi/KwModels/MNIST/small1/eval_"):
    file=dir+str(count)+".output"
    result = np.loadtxt(file)
    robust_err = result[:,3]
    err = result[:,5]
    return robust_err, err

def robust_voting(robust_err,clean_err):
    voting_robust_err = []
    voting_clean_err = []
    # for i in range(10000):
        # bot = 


if __name__ == "__main__": 
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--filename', default=opt)
    # args = parser.parse_args()

    # readFile(args.filename)
    # robust_err = []
    # clean_err = []
    # for i in [1,2,3,4,5,6]:
    #     r_err, c_err = readFile(count=i+1)
    #     robust_err.append(r_err)
    #     clean_err.append(c_err)
    # print(np.shape(robust_err),np.shape(clean_err))



    robust_err1, err1 = readFile(count=3)
    print(sum(robust_err1), sum(err1))
    print(sum(robust_err1<err1),sum(robust_err1>err1))