import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Visualize temperature data")
parser.add_argument("--exp", type=str, help="which experiment")
parser.add_argument("--dtm", type=str, help="which dtm")
args_parser = parser.parse_args()
exp = args_parser.exp
dtm = args_parser.dtm
all_temps = []
all_temps_logic = []
with open(f"visualization/{exp}/{dtm}/full_temperature_mem.trace","r") as f1:
    f1.readline()
    for line in f1:
        l = line.strip().split()
        all_temps_logic.append([float(x) for x in l[:16]])
        all_temps.append([float(x) for x in l[16:]])
all_temps = np.array(all_temps)
all_temps_logic = np.array(all_temps_logic)


# Generate example 2D data

# Bottom corner banks
bottom_corners = (all_temps[:,0]+all_temps[:,3]+all_temps[:,12]+all_temps[:,15])/4  # bottom corners
bottom_corners_peak = np.maximum.reduce([all_temps[:,0],all_temps[:,3],all_temps[:,12],all_temps[:,15]])
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(bottom_corners)), bottom_corners, label='Bottom corners average', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Bottom corners average')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/bottom_corners_temp_trace.png")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(bottom_corners_peak)), bottom_corners_peak, label='Bottom corners peak', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Bottom corners peak')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/bottom_cornerspeak_temp_trace.png")


# top corner banks
top_corners = (all_temps[:,112]+all_temps[:,115]+all_temps[:,127]+all_temps[:,124])/4  # top corners
#top corners peaks
top_corners_peak = np.maximum.reduce([all_temps[:,112],all_temps[:,115],all_temps[:,127],all_temps[:,124]])
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(top_corners)), top_corners, label='Top corners average', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Top corners average')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/top_corners_temp_trace.png")


plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(top_corners_peak)), top_corners_peak, label='Top corners peak', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Top corners peaks')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/top_cornerspeak_temp_trace.png")


# center banks
centers = (all_temps[:,53]+all_temps[:,54]+all_temps[:,57]+all_temps[:,58]+all_temps[:,69]+all_temps[:,70]+all_temps[:,73]+all_temps[:,74])/8  # centers
centers_peak = np.maximum.reduce([all_temps[:,53],all_temps[:,54],all_temps[:,57],all_temps[:,58],all_temps[:,69],all_temps[:,70],all_temps[:,73],all_temps[:,74]])  # centers

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(centers)), centers, label='Center average', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Center banks average')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/centers_temp_trace.png")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(centers_peak)), centers_peak, label='Center peak', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Center banks peak')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/centerspeak_temp_trace.png")





# bottom_center banks
bottom_centers = (all_temps[:,5]+all_temps[:,6]+all_temps[:,9]+all_temps[:,10])/4  #  bottom centers
bottom_centers_peak = np.maximum.reduce([all_temps[:,5],all_temps[:,6],all_temps[:,9],all_temps[:,10]])  #  bottom centers

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(bottom_centers)), bottom_centers, label='Bottom Center average', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Bottom Center banks average')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/bottom_centers_temp_trace.png")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(bottom_centers_peak)), bottom_centers_peak, label='Bottom Center peak', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Bottom Center banks peak')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/bottom_centerspeak_temp_trace.png")




# top center banks
top_centers = (all_temps[:,117]+all_temps[:,118]+all_temps[:,121]+all_temps[:,122])/4  # centers
top_centers_peak = np.maximum.reduce([all_temps[:,117],all_temps[:,118],all_temps[:,121],all_temps[:,122]])  # centers

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(top_centers)), top_centers, label='Top Center average', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Top Center banks average')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/top_centers_temp_trace.png")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(top_centers_peak)), top_centers_peak, label='Top Center peak', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Top Center banks peak')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/top_centerspeak_temp_trace.png")


# top edges banks
top_edge = (all_temps[:,113]+all_temps[:,114]+all_temps[:,116]+all_temps[:,119]+all_temps[:,120]+all_temps[:,123]+all_temps[:,125]+all_temps[:,126])/8  # centers
top_edge_peak = np.maximum.reduce([all_temps[:,113],all_temps[:,114],all_temps[:,116],all_temps[:,119],all_temps[:,120],all_temps[:,123],all_temps[:,125],all_temps[:,126]])  # centers

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(top_edge)), top_edge, label='top edges average', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Top edges banks average')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/top_edge_temp_trace.png")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(top_edge_peak)), top_edge_peak, label='top edges peak', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Top edges banks peak')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/top_edgepeak_temp_trace.png")

# bottom edges banks
bottom_edge = (all_temps[:,1]+all_temps[:,2]+all_temps[:,4]+all_temps[:,7]+all_temps[:,8]+all_temps[:,11]+all_temps[:,13]+all_temps[:,14])/8  # centers
bottom_edge_peak = np.maximum.reduce([all_temps[:,1],all_temps[:,2],all_temps[:,4],all_temps[:,7],all_temps[:,8],all_temps[:,11],all_temps[:,13],all_temps[:,14]])  # centers

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(bottom_edge)), bottom_edge, label='bottom edges average', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Bottom edges banks average')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/bottom_edge_temp_trace.png")

plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(bottom_edge_peak)), bottom_edge_peak, label='bottom edges peak', color='b', linestyle='-', marker='o', markersize=4)
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Temp (C)')
plt.title('Temp trace Bottom edges banks peak')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/bottom_edgepeak_temp_trace.png")


with open(f"visualization/{exp}/{dtm}/stalls.txt","r") as f1:
    tot_stalls = int( f1.readline().lstrip("Stall count: ").strip())
    stall_trace = []
    f1.readline()
    for line in f1:
        stall_trace.append(int(line.strip()))


plt.figure(figsize=(8, 6))
plt.scatter(np.arange(len(stall_trace)), stall_trace, label='stall trace', color='b',  marker='o')
#plt.ylim(top=90)
# Add labels and title
plt.xlabel('Time')
plt.ylabel('Stall count')
plt.title('Core stall count trace')
#plt.legend()
plt.grid(True)
plt.savefig(f"visualization/{exp}/{dtm}/stall_trace.png")

