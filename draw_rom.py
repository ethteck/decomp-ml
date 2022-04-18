import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from golddata import *


def to_hex(x, pos):
    return '%x' % int(x)


# game = ("Pokemon Snap (U)", snap_gold, 0xB00000)
# game = ("Lylat Wars", 0, 0)
game = ("Paper Mario", paper_mario_gold, 0)

with open("mario_preds.pkl", "rb") as f:
    preds = pickle.load(f)

fig, ax = plt.subplots(figsize=(8, 100))
#ax.set_ylim(0, game[2])
fmt = ticker.FuncFormatter(to_hex)
loc = ticker.MultipleLocator(base=0xB000)

ax.set_title(game[0] + ' code/data map')

ax.set_ylabel('File offset')
ax.get_yaxis().set_major_formatter(fmt)
ax.get_yaxis().set_major_locator(loc)
ax.invert_yaxis()

for prediction in preds[1:]:
    if prediction[2] == 1:
        ax.bar("Model", prediction[1] - prediction[0], bottom=prediction[0], label='CODE', color="blue")
    elif prediction[2] == 0:
        ax.bar("Model", prediction[1] - prediction[0], bottom=prediction[0], label='DATA', color="yellow")

for gold in game[1]:
    # todo actual label instead of gold
    if gold[2] == "asm":
        ax.bar("GOLD", gold[1] - gold[0], bottom=gold[0], label='CODE', color="blue")
    else:
        ax.bar("GOLD", gold[1] - gold[0], bottom=gold[0], label='DATA', color="yellow")


plt.savefig("test.png", bbox_inches='tight')
plt.show()
