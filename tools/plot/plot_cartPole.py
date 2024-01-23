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
parser.add_argument("-r", "--real_time", help = "Plot the result in real time",
                    action="store_true")
parser.add_argument("-m", "--mean", help = "The mean of the data",
                    action="store_true")
parser.add_argument("-pl", "--pre_learning", help="The num of the test we want to plot, with pre-learning",
                     action='append')
parser.add_argument("-ag", "--angle", help = "Use a specific angle",
                    action="store_true")
parser.add_argument("-us", "--use_stat", help = "Compute stat about inside/external use",
                    action="store_true")
parser.add_argument("-ns", "--n_state", help = "Number of states",
                    type=int)
parser.add_argument("-iter", "--max_iter", help = "Maximum of iteration",
                    type=int)
args = parser.parse_args()

print(">> START PLOT:")
print(">>     num test (meta state):", args.list_ms)
print(">>     num test (sac):", args.list_sac)
print(">>     pre-learning:", args.pre_learning)
print(">>     real time:", args.real_time)
print(">>     mean:", args.mean)
print(">>     angle:", args.angle)
print(">>     use stat:", args.use_stat)
print(">>     number of states:", args.n_state)
print(">>     max iteration:", args.max_iter)

add_name=""
if args.angle:
    add_name += "_angle"



if not args.use_stat:
    if not args.real_time and not args.mean:
        legend = []
        if args.list_ms is not None:
            data = []
            for i in range(len(args.list_ms)):
                data.append( {"Iteration": [], "Reward": []})
            for i in range(len(args.list_ms)):
                print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] + add_name + ".txt")
                with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] + add_name + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data[i]["Iteration"]+= loaded_data[1][:args.max_iter]
                    data[i]["Reward"]+= loaded_data[0][:args.max_iter]
                    fig = sns.lineplot(x="Iteration", y="Reward", data=data[i])
                    legend.append("Meta-states" + args.list_ms[i])

        if args.list_sac is not None:
            data_sac = []
            for i in range(len(args.list_sac)):
                data_sac.append( {"Iteration": [], "Reward": []})
            for i in range(len(args.list_sac)):
                print(">> Load: ./Results/CartPol/sac/rewards_"+ args.list_sac[i] +add_name + ".txt")
                with open("./Results/CartPol/sac/rewards_"+ args.list_sac[i] +add_name + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_sac[i]["Iteration"]+= loaded_data[1][:args.max_iter]
                    data_sac[i]["Reward"]+= loaded_data[0][:args.max_iter]
                    fig = sns.lineplot(x="Iteration", y="Reward", data=data_sac[i])
                    legend.append("SAC" + args.list_sac[i])

        if args.pre_learning is not None:
            data_pre_learning = []
            for i in range(len(args.pre_learning)):
                data_pre_learning.append( {"Iteration": [], "Reward": []})
            for i in range(len(args.pre_learning)):
                print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl" + add_name +  ".txt")
                with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl"+add_name +  ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_pre_learning[i]["Iteration"]+= loaded_data[1][:args.max_iter]
                    data_pre_learning[i]["Reward"]+= loaded_data[0][:args.max_iter]
                    fig = sns.lineplot(x="Iteration", y="Reward", data=data_pre_learning[i])
                    legend.append("Meta-states (pre-learning)" + args.pre_learning[i])


        fig.legend(tuple(legend))
        plt.show()


    elif args.real_time and not args.mean:
        print(">>    START REAL TIME VIEW")
        legend = []
        if args.list_ms is not None:
            data = []
            for i in range(len(args.list_ms)):
                data.append( {"Iteration": [], "Reward": []})

            plt.ion()
            plt.show()

            while True:
                for i in range(len(args.list_ms)):
                    print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] +add_name + ".txt")
                    with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] +add_name + ".txt", 'r') as fp:
                        loaded_data = json.load(fp)
                        data[i]["Iteration"]= loaded_data[1][:args.max_iter]
                        data[i]["Reward"]= loaded_data[0][:args.max_iter]
                        fig = sns.lineplot(x="Iteration", y="Reward", data=data[i])
                        legend.append("Test (meta-states) " + args.list_ms[i])

                fig.legend(tuple(legend))
                plt.draw()
                plt.pause(2)
                plt.clf()

        if args.pre_learning is not None:
            data_pre_learning = []
            for i in range(len(args.pre_learning)):
                data_pre_learning.append( {"Iteration": [], "Reward": []})

            plt.ion()
            plt.show()
            while True:
                for i in range(len(args.pre_learning)):
                    print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl"+add_name + ".txt")
                    with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl"+add_name +  ".txt", 'r') as fp:
                        loaded_data = json.load(fp)
                        data_pre_learning[i]["Iteration"]= loaded_data[1][:args.max_iter]
                        data_pre_learning[i]["Reward"]= loaded_data[0][:args.max_iter]

                        fig = sns.lineplot(x="Iteration", y="Reward", data=data_pre_learning[i])
                        legend.append("Test (meta-states pre-learning) " + args.pre_learning[i])

                fig.legend(tuple(legend))
                plt.draw()
                plt.pause(2)
                plt.clf()



        if args.list_sac is not None:
            data_sac = []
            for i in range(len(args.list_sac)):
                data_sac.append( {"Iteration": [], "Reward": []})

            plt.ion()
            plt.show()
            while True:
                for i in range(len(args.list_sac)):
                    print(">> Load: ./Results/CartPol/sac/rewards_"+ args.list_sac[i] +add_name + ".txt")
                    with open("./Results/CartPol/sac/rewards_"+ args.list_sac[i] +add_name + ".txt", 'r') as fp:
                        loaded_data = json.load(fp)
                        data_sac[i]["Iteration"]= loaded_data[1][:args.max_iter]
                        data_sac[i]["Reward"]= loaded_data[0][:args.max_iter]
                        fig = sns.lineplot(x="Iteration", y="Reward", data=data_sac[i])
                        legend.append("Test (sac)" + args.list_sac[i])

                fig.legend(tuple(legend))
                plt.draw()
                plt.pause(2)
                plt.clf()
    else:
        legend = []
        if args.list_ms is not None:
            data = {"Iteration": [], "Reward": []}
            for i in range(len(args.list_ms)):
                print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] +add_name + ".txt")
                with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] +add_name + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data["Iteration"]+= loaded_data[1][:args.max_iter]
                    data["Reward"]+= loaded_data[0][:args.max_iter]

            fig = sns.lineplot(x="Iteration", y="Reward", data=data)
            legend.append("Meta-states")

        if args.list_sac is not None:
            data_sac = {"Iteration": [], "Reward": []}
            for i in range(len(args.list_sac)):
                print(">> Load: ./Results/CartPol/sac/rewards_"+ args.list_sac[i] +add_name + ".txt")
                with open("./Results/CartPol/sac/rewards_"+ args.list_sac[i] +add_name + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_sac["Iteration"]+= loaded_data[1][:args.max_iter]
                    data_sac["Reward"]+= loaded_data[0][:args.max_iter]
            fig = sns.lineplot(x="Iteration", y="Reward", data=data_sac)
            legend.append("SAC")

        if args.pre_learning is not None:
            data_pre_learning = {"Iteration": [], "Reward": []}
            for i in range(len(args.pre_learning)):
                print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl"+add_name +  ".txt")
                with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl"+ add_name +".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    data_pre_learning["Iteration"]+= loaded_data[1][:args.max_iter]
                    data_pre_learning["Reward"]+= loaded_data[0][:args.max_iter]

            fig = sns.lineplot(x="Iteration", y="Reward", data=data_pre_learning)
            legend.append("Meta-states (pre-learning)")

        fig.legend(tuple(legend))
        plt.show()
else:
        legend = []
        if args.list_ms is not None:
            data= {"Iteration": [], "Inside": [], "External": []}
            for i in range(len(args.list_ms)):
                print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] +add_name + ".txt")
                with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.list_ms[i] +add_name + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    inside, external, tot = loaded_data[2][:args.max_iter]

                    data["Iteration"]+= tot[:args.max_iter]
                    data["Inside"]+= inside[:args.max_iter]
                    data["External"]+= external[:args.max_iter]

            fig = sns.lineplot(x="Iteration", y="Inside", data=data, ci=None)
            fig = sns.lineplot(x="Iteration", y="External", data=data, ci=None)

            legend.append("Meta-states - inside")
            legend.append("Meta-states - external")


        if args.pre_learning is not None:
            data_pre_learning = {"Iteration": [], "Inside": [], "External": []}

            for i in range(len(args.pre_learning)):
                print(">> Load: ./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl"+add_name + ".txt")
                with open("./Results/CartPol/v_n_state="+str(args.n_state)+"/rewards_"+ args.pre_learning[i] +"_pl"+ add_name + ".txt", 'r') as fp:
                    loaded_data = json.load(fp)
                    inside, external, tot = loaded_data[2][:args.max_iter]

                    data_pre_learning["Iteration"]+= tot[:]
                    data_pre_learning["Inside"]+= inside[:]
                    data_pre_learning["External"]+= external[:]

            fig = sns.lineplot(x="Iteration", y="Inside", data=data_pre_learning, ci=None)
            fig = sns.lineplot(x="Iteration", y="External", data=data_pre_learning, ci=None)

            legend.append("Meta-states (pre-learning) - inside")
            legend.append("Meta-states (pre-learning) - external")

        fig.legend(tuple(legend))
        plt.show()
