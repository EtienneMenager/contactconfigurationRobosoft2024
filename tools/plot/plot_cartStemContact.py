import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("-lm", "--list_ms", help="The num of the test we want to plot",
                     action='append')
parser.add_argument("-ls", "--list_sac", help="The num of the test we want to plot",
                     action='append')
parser.add_argument("-lq", "--list_qp", help="The num of the test we want to plot",
                     action='append')
parser.add_argument("-m", "--mean", help = "The mean of the data",
                    action="store_true")
parser.add_argument("-us", "--use_stat", help = "Compute stat about inside/external use",
                    action="store_true")
parser.add_argument("-iter", "--max_iter", help = "Maximum of iteration",
                    type=int)
args = parser.parse_args()

print(">> START PLOT:")
print(">>     num test (meta state):", args.list_ms)
print(">>     num test (sac):", args.list_sac)
print(">>     mean:", args.mean)
print(">>     use stat:", args.use_stat)
print(">>     max iteration:", args.max_iter)


if not args.use_stat:
    if not args.mean:
        legend = []
        if args.list_ms is not None:
            data = []
            for i in range(len(args.list_ms)):
                data.append( {"Iteration": [], "Reward": []})
            for i in range(len(args.list_ms)):
                print(">> Load: ./Results/CartStemContact/meta_states/rewards_"+ args.list_ms[i] + ".txt")
                with open("./Results/CartStemContact/meta_states/rewards_"+ args.list_ms[i] + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data[i]["Iteration"]+= loaded_data[1]
                    data[i]["Reward"]+= loaded_data[0]
                    fig = sns.lineplot(x="Iteration", y="Reward", data=data[i], ci = None)
                    legend.append("Meta-states" + args.list_ms[i])

        if args.list_sac is not None:
            data_sac = []
            for i in range(len(args.list_sac)):
                data_sac.append( {"Iteration": [], "Reward": []})
            for i in range(len(args.list_sac)):
                print(">> Load: ./Results/CartStemContact/sac/rewards_"+ args.list_sac[i] + ".txt")
                with open("./Results/CartStemContact/sac/rewards_"+ args.list_sac[i] + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_sac[i]["Iteration"]+= loaded_data[1]
                    data_sac[i]["Reward"]+= loaded_data[0]
                    fig = sns.lineplot(x="Iteration", y="Reward", data=data_sac[i], ci = None)
                    legend.append("SAC" + args.list_sac[i])

        if args.list_qp is not None:
            data_qp = []
            for i in range(len(args.list_qp)):
                data_qp.append( {"Iteration": [], "Reward": []})
            for i in range(len(args.list_qp)):
                print(">> Load: ./Results/CartStemContact/QP/rewards_"+ args.list_qp[i] + ".txt")
                with open("./Results/CartStemContact/QP/rewards_"+ args.list_qp[i] + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_qp[i]["Iteration"]+= loaded_data[1]
                    data_qp[i]["Reward"]+= loaded_data[0]
                    fig = sns.lineplot(x="Iteration", y="Reward", data=data_qp[i], ci = None)
                    legend.append("QP" + args.list_qp[i])

        fig.legend(tuple(legend))
        plt.show()

    else:
        legend = []
        if args.list_ms is not None:
            data = {"Iteration": [], "Reward": []}
            for i in range(len(args.list_ms)):
                print(">> Load: ./Results/CartStemContact/meta_states/rewards_"+ args.list_ms[i]+ ".txt")
                with open("./Results/CartStemContact/meta_states/rewards_"+ args.list_ms[i] + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data["Iteration"]+= loaded_data[1]
                    print(args.list_ms[i], ":", loaded_data[1][-1])
                    data["Reward"]+= loaded_data[0]

            fig = sns.lineplot(x="Iteration", y="Reward", data=data, ci = None)
            legend.append("Meta-states")

        if args.list_sac is not None:
            data_sac = {"Iteration": [], "Reward": []}
            for i in range(len(args.list_sac)):
                print(">> Load: ./Results/CartStemContact/sac/rewards_"+ args.list_sac[i] + ".txt")
                with open("./Results/CartStemContact/sac/rewards_"+ args.list_sac[i] + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_sac["Iteration"]+= loaded_data[1]
                    data_sac["Reward"]+= loaded_data[0]
            fig = sns.lineplot(x="Iteration", y="Reward", data=data_sac, ci = None, color = 'orange')
            legend.append("SAC")

        if args.list_qp is not None:
            data_qp = {"Iteration": [], "Reward": []}
            for i in range(len(args.list_qp)):
                print(">> Load: ./Results/CartStemContact/QP/rewards_"+ args.list_qp[i] + ".txt")
                with open("./Results/CartStemContact/QP/rewards_"+ args.list_qp[i] + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_qp["Iteration"]+= loaded_data[1]
                    data_qp["Reward"]+= loaded_data[0]
            fig = sns.lineplot(x="Iteration", y="Reward", data=data_qp, ci = None, color = 'purple')
            legend.append("QP")

        fig.legend(tuple(legend))
        plt.xlim(0, args.max_iter)
        plt.show()
else:
        legend = []
        if args.list_ms is not None:
            data= {"Iteration": [], "Inside": [], "External": []}
            for i in range(len(args.list_ms)):
                print(">> Load: ./Results/CartStemContact/meta_states/rewards_"+ args.list_ms[i] + ".txt")
                with open("./Results/CartStemContact/meta_states/rewards_"+ args.list_ms[i] +  ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    inside, external, tot = loaded_data[2][:args.max_iter]

                    data["Iteration"]+= tot
                    data["Inside"]+= inside
                    data["External"]+= external


            fig = sns.lineplot(x="Iteration", y="Inside", data=data, ci=None)
            fig = sns.lineplot(x="Iteration", y="External", data=data, ci=None)

            legend.append("Meta-states - inside")
            legend.append("Meta-states - external")

        fig.legend(tuple(legend))
        plt.xlim(0, args.max_iter)
        plt.show()
