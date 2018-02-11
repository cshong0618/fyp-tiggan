import matplotlib.pyplot as plt
import argparse

def plot_from_log(log_file):
    d_loss = []
    g_loss = []

    with open(log_file) as log:
        for l in log:
            if l[0] == 'd' or l[0] == 'g':
                ss = l.split("[")
                data = ss[1].split("]")[0].split(",")

                if l[0] == 'd':
                    for d in data:
                        d_loss.append(float(d))
                elif l[0] == 'g':
                    for d in data:
                        g_loss.append(float(d))

    if len(d_loss) != len(g_loss):
        raise Exception("d_loss [%d] and g_loss [%d] are of different length." % (len(d_loss), len(g_loss)))
    
    X = [i + 1 for i in range(len(d_loss))]

    D = plt.plot(X, d_loss, label="D")
    G = plt.plot(X, g_loss, label="G")

    plt.legend(["D", "G"])
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-L', '--log', default='', help="Log file name", required=True)
    args = parser.parse_args()

    log_file = args.log

    plot_from_log(log_file)