import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# read tsv file
def read_tsv(trainning_data: str) -> pd.DataFrame:
    return pd.read_csv(trainning_data,sep="\t",header=None)

def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf
# calc T -> T = (dH - dG)/dS
def formula_test(df: pd.DataFrame):
    df[4] = (df[2] - df[1])/df[3]
    mean = df[4].mean()
    std = df[4].std()
    print(mean)
    print(std)
    return
    x = np.arange(340, 380, .1)
    y = normfun(x, mean, std)
    plt.plot(x, y)
    # 绘制数据集的直方图
    plt.hist(df[4], bins=20, rwidth=0.6, density=True)
    plt.title('T distribution')
    plt.xlabel('T')
    plt.ylabel('Frequency')
    # 输出正态分布曲线和直方图
    plt.savefig("./figure.png")

def main() -> None:
    df = read_tsv("/data/ntc/Repository/thermodynamics/trainning_data_nocor.tsv")
    formula_test(df)
    
    


if __name__ == "__main__":
    main()
