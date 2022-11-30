from numpy import *
import numpy as np
import csv

# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):       # 转换数据类型
            row[i] = float(row[i])
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def MyBinaryMatrixToNum(DiseaseAndRNABinary):
    LncDiseaseNum = []
    counter = 0
    while counter < len(DiseaseAndRNABinary):
        counter1 = 0
        while counter1 < len(DiseaseAndRNABinary[counter]):
            if DiseaseAndRNABinary[counter][counter1] == 1:
                pair = []
                pair.append(counter1)  # 名为LncDiseaseNum！
                pair.append(counter)
                LncDiseaseNum.append(pair)
            counter1 = counter1 + 1
        counter = counter + 1
    return LncDiseaseNum

def partition(ls, size):
    """
    Returns a new list with elements
    of which is a list of certain size.

        >>> partition([1, 2, 3, 4], 3)
        [[1, 2, 3], [4]]
    """
    return [ls[i:i+size] for i in range(0, len(ls), size)]



def MyNewDiseaseAndRNABinary(TestList, TrainList, AllDisease, AllRNA, LncDiseaseNum):
    # 生成全为-1的矩阵
    NewDiseaseAndRNABinary = []
    counter = 0
    while counter < len(AllDisease):
        row = []
        counter1 = 0
        while counter1 < len(AllRNA):
            row.append(0)
            counter1 = counter1 + 1
        NewDiseaseAndRNABinary.append(row)
        counter = counter + 1
    # 向矩阵中填0和1
    for i in range(len(TestList)):
        NewDiseaseAndRNABinary[LncDiseaseNum[TestList[i]][1]][LncDiseaseNum[TestList[i]][0]] = 0
    for i in range(len(TrainList)):
        NewDiseaseAndRNABinary[LncDiseaseNum[TrainList[i]][1]][LncDiseaseNum[TrainList[i]][0]] = 1
    return NewDiseaseAndRNABinary




def GaussianKernelDisease(DiseaseAndRNABinary):
    # 计算rd
    counter1 = 0
    sum1 = 0
    while counter1 < (len(DiseaseAndRNABinary)):
        counter2 = 0
        while counter2 < (len(DiseaseAndRNABinary[counter1])):
            sum1 = sum1 + pow((DiseaseAndRNABinary[counter1][counter2]), 2)
            counter2 = counter2 + 1
        counter1 = counter1 + 1
    print('sum1=', sum1)
    Ak = sum1
    Nd = len(DiseaseAndRNABinary)
    rdpie = 0.5
    rd = rdpie * Nd / Ak
    print('disease rd', rd)
    # 生成DiseaseGaussian
    DiseaseGaussian = []
    counter1 = 0
    while counter1 < len(DiseaseAndRNABinary):  # 计算疾病counter1和counter2之间的similarity
        counter2 = 0
        DiseaseGaussianRow = []
        while counter2 < len(DiseaseAndRNABinary):  # 计算Ai*和Bj*
            AiMinusBj = 0
            sum2 = 0
            counter3 = 0
            AsimilarityB = 0
            while counter3 < len(DiseaseAndRNABinary[counter2]):  # 疾病的每个属性分量
                sum2 = pow((DiseaseAndRNABinary[counter1][counter3] - DiseaseAndRNABinary[counter2][counter3]),2)  # 计算平方
                AiMinusBj = AiMinusBj + sum2
                counter3 = counter3 + 1
            AsimilarityB = math.exp(- (AiMinusBj / rd))
            DiseaseGaussianRow.append(AsimilarityB)
            counter2 = counter2 + 1
        DiseaseGaussian.append(DiseaseGaussianRow)
        counter1 = counter1 + 1
        print(counter1)
    return DiseaseGaussian

def GaussianKernelRNA(DiseaseAndRNABinary):
    MDiseaseAndRNABinary = np.array(DiseaseAndRNABinary)  # 列表转为矩阵
    RNAAndDiseaseBinary = MDiseaseAndRNABinary.T  # 转置DiseaseAndMiRNABinary
    RNAGaussian = []
    counter1 = 0
    sum1 = 0
    while counter1 < (len(RNAAndDiseaseBinary)):  # rna数量
        counter2 = 0
        while counter2 < (len(RNAAndDiseaseBinary[counter1])):  # disease数量
            sum1 = sum1 + pow((RNAAndDiseaseBinary[counter1][counter2]), 2)
            counter2 = counter2 + 1
        counter1 = counter1 + 1
    print('sum1=', sum1)
    Ak = sum1
    Nm = len(RNAAndDiseaseBinary)
    rdpie = 0.5
    rd = rdpie * Nm / Ak
    print('RNA rd', rd)
    # 生成RNAGaussian
    counter1 = 0
    while counter1 < len(RNAAndDiseaseBinary):  # 计算rna counter1和counter2之间的similarity
        counter2 = 0
        RNAGaussianRow = []
        while counter2 < len(RNAAndDiseaseBinary):  # 计算Ai*和Bj*
            AiMinusBj = 0
            sum2 = 0
            counter3 = 0
            AsimilarityB = 0
            while counter3 < len(RNAAndDiseaseBinary[counter2]):  # rna的每个属性分量
                sum2 = pow((RNAAndDiseaseBinary[counter1][counter3] - RNAAndDiseaseBinary[counter2][counter3]),2)  # 计算平方，有问题？？？？？
                AiMinusBj = AiMinusBj + sum2
                counter3 = counter3 + 1
            AsimilarityB = math.exp(- (AiMinusBj / rd))
            RNAGaussianRow.append(AsimilarityB)
            counter2 = counter2 + 1
        RNAGaussian.append(RNAGaussianRow)
        counter1 = counter1 + 1
        print(counter1)
    return RNAGaussian

def TestGenerate(TestList,LncDiseaseNum,RNAGaussian, DiseaseGaussian):
    TestSampleFeature = []
    counter = 0
    while counter < len(TestList):
        FeaturePair = []
        FeaturePair.extend(RNAGaussian[LncDiseaseNum[TestList[counter]][0]])
        FeaturePair.extend(DiseaseGaussian[LncDiseaseNum[TestList[counter]][1]])
        TestSampleFeature.append(FeaturePair)
        counter = counter + 1
    return TestSampleFeature

def MySampleLabel(SampleFeature):
    # 生成SampleLabel。
    SampleLabel = []
    counter = 0
    while counter < len(SampleFeature) / 2:
        SampleLabel.append(1)
        counter = counter + 1
    counter1 = 0
    while counter1 < len(SampleFeature) / 2:
        SampleLabel.append(0)
        counter1 = counter1 + 1
    return SampleLabel


def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter][1])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb

def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction

def NegativeNumGenerate(LncDisease, AllDisease,AllRNA):
    # 负样本为全部的disease-rna（328*881）中随机抽取，未在内LncDisease即为负样本
    import random
    NegativeSample = []
    counterN = 0
    while counterN < len(LncDisease):  # 随机选出一个疾病rna对
        counterD = random.randint(0, len(AllDisease) - 1)
        counterR = random.randint(0, len(AllRNA) - 1)
        DiseaseAndRnaPair = []
        DiseaseAndRnaPair.append(AllRNA[counterR])
        DiseaseAndRnaPair.append(AllDisease[counterD])
        flag1 = 0
        counter = 0
        while counter < len(LncDisease):
            if DiseaseAndRnaPair == LncDisease[counter]:
                flag1 = 1
                break
            counter = counter + 1
        if flag1 == 1:
            continue
        flag2 = 0
        counter1 = 0
        while counter1 < len(NegativeSample):  # 在已选的负样本中没有，防止重复
            if DiseaseAndRnaPair == NegativeSample[counter1]:
                flag2 = 1
                break
            counter1 = counter1 + 1
        if flag2 == 1:
            continue
        if (flag1 == 0 & flag2 == 0):
            NamePair = []  # 生成对
            NamePair.append(counterR)
            NamePair.append(counterD)
            NegativeSample.append(NamePair)

            counterN = counterN + 1
    return NegativeSample

def NegativeFeatureGenerate(NegativeSampleNum, TrainList,TestList,GaussianKernelDisease, GaussianKernelRNA):
    TrainNegativeFeature = []
    for i in range(len(TrainList)):
        counterR = NegativeSampleNum[TrainList[i]][0]
        counterD = NegativeSampleNum[TrainList[i]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        TrainNegativeFeature.append(FeaturePair)

    TestNegativeFeature = []
    for j in range(len(TestList)):
        counterR = NegativeSampleNum[TestList[j]][0]
        counterD = NegativeSampleNum[TestList[j]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        TestNegativeFeature.append(FeaturePair)
    return TrainNegativeFeature, TestNegativeFeature

def PositiveFeatureGenerate(LncDiseaseNum, TrainList,TestList,GaussianKernelDisease, GaussianKernelRNA):
    TrainPositiveFeature = []
    for i in range(len(TrainList)):
        counterR = LncDiseaseNum[TrainList[i]][0]
        counterD = LncDiseaseNum[TrainList[i]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        TrainPositiveFeature.append(FeaturePair)

    TestPositiveFeature = []
    for j in range(len(TestList)):
        counterR = LncDiseaseNum[TestList[j]][0]
        counterD = LncDiseaseNum[TestList[j]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        TestPositiveFeature.append(FeaturePair)
    return TrainPositiveFeature, TestPositiveFeature


def NegativeFeatureGenerate2(NegativeSampleNum, TrainList,TestList,GaussianKernelDisease, GaussianKernelRNA, MyLncKmer, MyMiKmer):
    TrainNegativeFeature = []
    for i in range(len(TrainList)):
        counterR = NegativeSampleNum[TrainList[i]][0]
        counterD = NegativeSampleNum[TrainList[i]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        FeaturePair.extend(MyMiKmer[counterR])
        FeaturePair.extend(MyLncKmer[counterD])
        TrainNegativeFeature.append(FeaturePair)

    TestNegativeFeature = []
    for j in range(len(TestList)):
        counterR = NegativeSampleNum[TestList[j]][0]
        counterD = NegativeSampleNum[TestList[j]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        FeaturePair.extend(MyMiKmer[counterR])
        FeaturePair.extend(MyLncKmer[counterD])
        TestNegativeFeature.append(FeaturePair)
    return TrainNegativeFeature, TestNegativeFeature

def PositiveFeatureGenerate2(LncDiseaseNum, TrainList,TestList,GaussianKernelDisease, GaussianKernelRNA, MyLncKmer, MyMiKmer):
    TrainPositiveFeature = []
    for i in range(len(TrainList)):
        counterR = LncDiseaseNum[TrainList[i]][0]
        counterD = LncDiseaseNum[TrainList[i]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        FeaturePair.extend(MyMiKmer[counterR])
        FeaturePair.extend(MyLncKmer[counterD])
        TrainPositiveFeature.append(FeaturePair)

    TestPositiveFeature = []
    for j in range(len(TestList)):
        counterR = LncDiseaseNum[TestList[j]][0]
        counterD = LncDiseaseNum[TestList[j]][1]
        FeaturePair = []
        FeaturePair.extend(GaussianKernelRNA[counterR])
        FeaturePair.extend(GaussianKernelDisease[counterD])
        FeaturePair.extend(MyMiKmer[counterR])
        FeaturePair.extend(MyLncKmer[counterD])
        TestPositiveFeature.append(FeaturePair)
    return TrainPositiveFeature, TestPositiveFeature