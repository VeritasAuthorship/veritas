import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os




def plot_all(*files):
    plt.xlabel("# Epochs")
    plt.ylabel("Total Epoch Loss")

    for file in files:
        with open("plots/" + file, "rb") as f:
            history, _, _ = pickle.load(f)

        model = file.split("_")[0]
        data = file.split("_")[2]
        options = file.split("_")[3]

        plt.plot(list(range(len(history))), history, label=model + ", " + options)

    plt.legend()
    plt.show()

plot_all("LSTM_ATTN_SPOOKY__2018-12-14 17_00_47_199175.pickle", "LSTM_ATTN_SPOOKY_POS_2018-12-14 17_29_14_816444.pickle")#,"LSTM_ATTN_SPOOKY_POS_2018-12-14 18_11_57_447594.pickle")
# plot_all("LSTM_GUTENBERG__2018-12-14 18_50_19_901696.pickle")
# plot_all("LSTM_GUTENBERG__2018-12-14 18_50_19_901696.pickle", "LSTM_ATTN_GUTENBERG_POS_2018-12-14 07_24_27_891599.pickle")

# for file in os.listdir('plots'):
#     try:
#         with open("plots/" + file, "rb") as f:
#             history, _, _ = pickle.load(f)
#             print(file)
#             # plt.ylim(bottom=0)
#             # plt.xlim(left=0)
#             plt.xlabel("# Epochs")
#             plt.xlabel("Total Epoch Loss")
#             plt.plot(list(range(len(history))), history)
#             plt.show()
#     except Exception:
#         print(file + "failed.")