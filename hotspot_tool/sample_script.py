"""
Script to see heat spread trend in 3D memory

"""

import sys, os
import time as exetime
import argparse
import random
import numpy as np

bank_size = 64  # Mb
no_columns = 1                                      # in Kilo
no_bits_per_column = 8                              # 8 bits per column. Hence 8Kb row buffer.
no_rows = bank_size/no_columns/no_bits_per_column    # in Kilo, number of rows per bank
energy_per_read_access = 20.55
energy_per_write_access = 20.55
#logic_core_power = 0.272
logic_core_power = 0.0
energy_per_refresh_access = 3.55
sampling_interval = 1000000    #time in ns
interval_sec = sampling_interval * 1e-9
timestep = sampling_interval/1000                       # in uS. Should be in sync with hotspot.config (sampling_intvl)
t_refi = 7.8
no_refesh_commands_in_t_refw = 8
rows_refreshed_in_refresh_interval = no_rows/no_refesh_commands_in_t_refw  # for 512Mb bank, 8 rows per refresh => for 64Mb bank, 1 rows per refresh
bank_static_power = 0

type_of_stack = '3Dmem'
channel_buswidth = 128 # in bits
pseudochannel_buswidth = 64 # in bits
clock = 2 #GHz   can go max 3.2 GHz
time_period = 0.5 # time in ns
per_channel_size = bank_size * 16 * 1e6 # in bits
migration_access_count =  per_channel_size//pseudochannel_buswidth
clock_cycles_required = migration_access_count // 2 # since 2 banks are accessed in parallel
migration_time = time_period * clock_cycles_required # in ns
interval_counts_per_migration = migration_time//sampling_interval + 1
#migration_access_per_interval_per_bank = ((sampling_interval // time_period)*2 ) // 16
migration_access_per_interval_per_bank = migration_access_count // interval_counts_per_migration //16
# Bank Floorplan info
banks_in_x = 4
banks_in_y = 4
banks_in_z = 8
NUM_BANKS = 128

# Logic Floorplan info (only for 3Dmem)
logic_cores_in_x = banks_in_x
logic_cores_in_y = banks_in_y
NUM_LC = logic_cores_in_x * logic_cores_in_y
sniper_root = "/media/sayan/user_data/Documents/iitd_resources/Fourth/COL719/Project/COL719_assignment_comet"
# Configuring hotspot tool path
#hotspot_path = os.path.join(os.getenv('SNIPER_ROOT'), 'hotspot_tool/')
hotspot_path = os.path.join(sniper_root, 'hotspot_tool/')

executable = hotspot_path + 'hotspot'

parser = argparse.ArgumentParser(description="Synthetic access rate in 3D memory to see heat spread")
parser.add_argument("--board_config_file", type=str, help="Architecture configuration 3Dmem_16core for TSV and 3Dmem_16core_mono for MIV")
args_parser = parser.parse_args()

# hotspot_config_path = os.getenv('SNIPER_ROOT') + '/' 
hotspot_config_path = sniper_root + '/'
init_file_external = hotspot_config_path + 'config/hotspot/' + args_parser.board_config_file + '/mem.init'
hotspot_config_file = hotspot_config_path + 'config/hotspot/' + args_parser.board_config_file + '/mem_hotspot.config'
hotspot_floorplan_folder = hotspot_config_path + 'config/hotspot/' + args_parser.board_config_file
hotspot_layer_file = hotspot_config_path + 'config/hotspot/' + args_parser.board_config_file + '/mem.lcf'

# Output Parameters for hotspot simulation
# memory related files
hotspot_steady_temp_file = 'steady_temperature_mem.log'
hotspot_grid_steady_file = 'grid_steady_mem.log'
hotspot_all_transient_file = 'all_transient_mem.init'
power_trace_file = 'power_mem.trace'
power_trace_file_total = 'tmmpFile_power2'
full_power_trace_file = 'full_power_mem.trace'
full_totalpower_trace_file = 'full_totalpower_mem.trace'
temperature_trace_file = 'temperature_mem.trace'
full_temperature_trace_file = 'full_temperature_mem.trace'
init_file = 'temperature_mem.init'

# Invoke hotspot to generate temperature trace for the corresponding power trace. 
# The generated transient temperature trace (all_transient_file) is used as an init file for the next iteration

hotspot_command = executable  \
                  + ' -c ' + hotspot_config_file \
                  + ' -p ' + power_trace_file \
				  + ' -pTot ' + power_trace_file_total \
                  + ' -o ' + temperature_trace_file \
                  + ' -model_secondary 1 -model_type grid ' \
                  + ' -steady_file ' + hotspot_steady_temp_file \
                  + ' -all_transient_file ' + hotspot_all_transient_file \
                  + ' -grid_steady_file ' + hotspot_grid_steady_file \
                  + ' -steady_state_print_disable 1 ' \
                  + ' -l 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, ' \
                  + ' -type ' + type_of_stack \
                  + ' -sampling_intvl ' + str(interval_sec) \
                  + ' -grid_layer_file ' + hotspot_layer_file \
                  + ' -detailed_3D on'
#                  + ' -f ' + hotspot_floorplan_file \

print("Hotspot command being used:\n",hotspot_command)

#initialization and setting up files
os.system("echo copying files for first run")
if (init_file_external != "None"):
    os.system("ls -l " + init_file_external)
    os.system("cp " + init_file_external + " " + init_file)
#os.system('mkdir -p hotspot')
#os.system("cp -r " + hotspot_floorplan_folder + " " + './hotspot')
os.system("rm -f " + full_temperature_trace_file)
os.system("rm -f " + full_power_trace_file)

def gen_mem_header():
    """
    Return header for memory banks.
    Similar to gen_ptrace_header, but will not write to file and will not include other components than memory.
    """
    # For a 2by1 ptrace with four layers header should be B0_0    B0_1    B0_0    B0_1    B0_0    B0_1    B0_0    B0_1
    # For 3D: core is at the top, whereas for 3Dmem, 2.5D config. the 3Dmem is at the bottom.
    # Creating Header
    mem_header = ''

    for b in range(NUM_BANKS):
      mem_header = mem_header + "B_" + str(b) + "\t"

    return mem_header

def gen_ptrace_header():
    # For a 2by1 ptrace with four layers header should be B0_0    B0_1    B0_0    B0_1    B0_0    B0_1    B0_0    B0_1
    # For 3D: core is at the top, whereas for 3Dmem, 2.5D config. the 3Dmem is at the bottom.
    # Creating Header
    ptrace_header = ''
    
    if type_of_stack== "3Dmem":
        for x in range(0,logic_cores_in_x):
            for y in range(0,logic_cores_in_y):
                ptrace_header=ptrace_header + "LC_" + str(x*logic_cores_in_y + y) + "\t" 
    
    for z in range(0,banks_in_z):
        for x in range(0,banks_in_x):
            for y in range(0,banks_in_y):
                bank_number = z*banks_in_x*banks_in_y + x*banks_in_y + y
                ptrace_header=ptrace_header + "B_" + str(bank_number) + "\t" 
    
    with open("%s" %(power_trace_file), "w") as f:
        f.write("%s\n" %(ptrace_header))
    #f.close()
    return ptrace_header

def writeBankStateFile(bankstate):
    banks_per_layer = NUM_BANKS // banks_in_z  # Banks in each layer (4x4 grid = 16)
    with open("BankStateData.txt", "w") as f:
        for i in range(0, NUM_BANKS, banks_per_layer):
            line = " ".join(str(x) for x in bankstate[i:i+banks_per_layer])
            f.write(line + "\n")

# The main portion which invokes hotspot to generate temperature.trace
class memTherm:
  def setup(self, args):
    args = dict(enumerate((args or '').split(':')))
    filename = args.get(0, None)
    interval_ns = sampling_interval
    # print interval_ns 
   
    self.bankstate = []
    self.lpm_pseudochannels = set()
    self.lpm_channels = set()
    self.migration_activity = [0 for i in range(16)]
    self.idle_pseudochannels = set()
    if filename:
      self.fd = open(os.path.join(hotspot_path, filename), 'w')
      self.isTerminal = False
    else:
      self.fd = sys.stdout
      self.isTerminal = True

    self.stats = {
      'stat_rd': [0 for bank in range(NUM_BANKS)],
      'stat_wr': [0 for bank in range(NUM_BANKS)],
    }

    banks_per_layer = NUM_BANKS // banks_in_z  # Banks in each layer (4x4 grid = 16)
    all_banks = [[layer * banks_per_layer + bank for bank in range(banks_per_layer)] for layer in range(banks_in_z)]

    for layer in range(banks_in_z):
        for bank in range((layer) * banks_per_layer, (layer + 1) * banks_per_layer):
            self.bankstate.append(1)

    writeBankStateFile(self.bankstate)
    # print the initial header into different log/trace files
    #gen_ptrace_header()
    ptrace_header = gen_ptrace_header()
    with open(full_temperature_trace_file, "w") as f:
        f.write("%s\n" %(ptrace_header))
    f.close()
    with open(full_power_trace_file, "w") as f:
        f.write("%s\n" %(ptrace_header))
    f.close()
    with open(full_totalpower_trace_file, "w") as f:
        f.write("%s\n" %(ptrace_header))
    f.close()

  # return access rates of various memory banks
  def get_access_rates(self, time):
    access_rates_read = [0 for number in range(NUM_BANKS)]
    for bank in range(NUM_BANKS):
      statdiff_rd = self.stats['stat_rd'][bank]
      access_rates_read[bank] = statdiff_rd
      self.fd.write(' %u' % statdiff_rd)
    self.fd.write('\n')
#    print access_rates
    access_rates_write = [0 for number in range(NUM_BANKS)]
    for bank in range(NUM_BANKS):
      statdiff_wr = self.stats['stat_wr'][bank]
      access_rates_write[bank] = statdiff_wr
      self.fd.write(' %u' % statdiff_wr)
    self.fd.write('\n');self.fd.write('\n')

    return access_rates_read, access_rates_write

    # calculate power trace using access rate and other parameters
  def calc_power_trace(self, time):
    accesses_read, accesses_write = self.get_access_rates(time)
 #    print accesses 

    avg_no_refresh_intervals_in_timestep = timestep/t_refi                                                     # 20/7.8 = 2.56 refreshes on an average 
    avg_no_refresh_rows_in_timestep = avg_no_refresh_intervals_in_timestep * rows_refreshed_in_refresh_interval # 2.56*8 rows refreshed = 20.48 refreshes 
    refresh_energy_in_timestep = avg_no_refresh_rows_in_timestep * energy_per_refresh_access                   # 20.48 * 100 nJ = 2048 nJ, 100 nJ (say) is the energy per refresh access
    avg_refresh_power = refresh_energy_in_timestep/(timestep * 1000)
    bank_power_trace = [0 for number in range(NUM_BANKS)]
    # calculate bank power for each bank using access traces

    for bank in range(NUM_BANKS):
        bank_power_trace[bank] = self.bankstate[bank]*(accesses_read[bank] * energy_per_read_access + accesses_write[bank] * energy_per_write_access)/(timestep*1000) \
                      + bank_static_power + avg_refresh_power
        bank_power_trace[bank] = round(bank_power_trace[bank], 3)
    logic_power_trace = []
    # create logic_core power array. applicable only for 3Dmem memory
    if (type_of_stack == "3Dmem"):
        logic_power_trace = [logic_core_power for number in range(NUM_LC)]
    power_trace = ''
    # convert power trace into a concatenated string for formated output
    # print logic power trace to the main power_trace
    for p in logic_power_trace:
        power_trace = power_trace + str(p) + '\t'
    # add bank power into the main power trace
    for bank in range(len(bank_power_trace)):
        # add bank power trace for all type of memories
        power_trace = power_trace + str(bank_power_trace[bank]) + '\t'
    power_trace = power_trace + "\r\n"
    ptrace_header = gen_ptrace_header()
    # write power information into the trace file for use by hotspot
    with open("%s" %(power_trace_file), "w") as f:
        f.write("%s\n" %(ptrace_header))
        f.write("%s" %(power_trace))
        #print(power_trace)
    #f.close()
    return power_trace

  # invokes hotspot to generate the temperature trace
  def calc_temperature_trace(self, time):
    # print power_trace
    # calculate memory power trace (combines with core trace in case of 3D and 2.5D within function)
    self.calc_power_trace(time)
    # invoke the memory hotspot. It will include core parts automatically for 3D and 2.5D
    hcmd = hotspot_command
    hcmd += ' -v ' + '1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1'
    first_run = (sum(1 for linee in open(full_temperature_trace_file, 'r')) == 1) 
    if (init_file_external != "None") or (not first_run):
        hcmd += ' -init_file ' + init_file
    start_t = exetime.time()
    os.system(hcmd)
    end_t = exetime.time()
    print("[Time] Total hotspot runtime:", end_t - start_t)

    os.system("cp " + hotspot_all_transient_file + " " + init_file)
    os.system("tail -1 " + temperature_trace_file + ">>" + full_temperature_trace_file)
    os.system("tail -1 " + power_trace_file + " >>" + full_power_trace_file)
    os.system("tail -1 " + power_trace_file_total + " >>" + full_totalpower_trace_file)


"""
Call calc_temp_trace periodically 200 times
"""

t1 = memTherm()
t1.setup("access_rates.txt")
# Changing read/write access rate to 1 (constant)
# for i in range(112,128):
#    #if (i//16)%2==0 :
#   t1.stats['stat_rd'][i] = 1000000
#   t1.stats['stat_wr'][i] = 1000000

# for t in range(100):
#     if t%10 == 0:
#       for i in range(NUM_BANKS):
#         t1.stats['stat_rd'][i] = int(t1.stats['stat_rd'][i]**0.5-1000);t1.stats['stat_rd'][i]*=t1.stats['stat_rd'][i]
#         t1.stats['stat_wr'][i] = int(t1.stats['stat_wr'][i]**0.5-1000);t1.stats['stat_wr'][i]*=t1.stats['stat_wr'][i]
#     t1.calc_temperature_trace(t)
# #print(hotspot_floorplan_folder,hotspot_layer_file,hotspot_path)


# Useful functions
def bank_to_pseudochannel(bank):  # gives the pseudochannel number for given bank
   if bank>=64 : bank-=64
   return bank//4

def bank_to_channel(bank):  # gives the channel number for given bank
   if bank>=64 : bank-=64
   return bank//8

def pseudochannel_to_banks(pchannel):  # gives banks of given pseudochannel
   banks = [i for i in range(pchannel*4,pchannel*4+4)]
   banks += [i for i in range(pchannel*4+64,pchannel*4+68)]
   return banks

def channel_to_banks(channel):  # gives banks of given channel
   banks = [i for i in range(channel*8,channel*8+8)]
   banks += [i for i in range(channel*8+64,channel*8+72)]
   return banks

def get_temp(temp_file): # returns current temperature of all banks
  with open(temp_file,"r") as f1:
     f1.readline()
     temps = [float(x) for x in f1.readline().strip().split()[16:]]
  return np.array(temps)

def get_power(power_file):
   with open(power_file,"r") as f1:
       f1.readline()
       power = [float(x) for x in f1.readline().strip().split()[16:]]
   return np.array(power)

def get_temp_pseudochannels(temp_files):
   temps = get_temp(temp_files)
   pchannel_temps = []
   for c in range(16):
      pchannel_temps.append(np.max(temps[pseudochannel_to_banks(c)]))
   return np.array(pchannel_temps)

def get_power_channels(power_files):
   power = get_power(power_files)
   channel_powers = []
   for c in range(8):
      channel_powers.append(np.sum(power[channel_to_banks(c)]))
   return np.array(channel_powers)

def get_temp_channels(temp_files):
   temps = get_temp(temp_files)
   channel_temps = []
   for c in range(8):
      channel_temps.append(np.max(temps[channel_to_banks(c)]))
   return np.array(channel_temps)


def banks_reached_threshold(temps,threshold):
  banks = []
  for i in range(NUM_BANKS):
    if temps[i] >= threshold:
      banks.append(i)
  return banks

def pseudochannels_reached_threshold(temps,threshold,cool_down):
   channels = []
   t = True
   cool_channels = []
   for i in range(8):
      t = True
      for bank in pseudochannel_to_banks(i):
         if (temps[bank]>cool_down):
            if (temps[bank]>threshold):
               channels.append(i)
               t = False
               break
            t = False
      if t : cool_channels.append(i)
         
   return channels,cool_channels

def channels_reached_threshold(temps,threshold,cool_down):
   channels = []
   t = True
   cool_channels = []
   for i in range(8):
      t = True
      for bank in channel_to_banks(i):
         if (temps[bank]>cool_down):
            if (temps[bank]>threshold):
               channels.append(i)
               t = False
               break
            t = False
      if t : cool_channels.append(i)
         
   return channels,cool_channels

def channels_reached_threshold_layerwise(temps,thresholds,cool_down):
   channels = []
   t = True
   cool_channels = []
   for i in range(8):
      t = True
      for bank in channel_to_banks(i):
         if (temps[bank]>cool_down[bank//16]):
            if (temps[bank]>thresholds[bank//16]):
               channels.append(i)
               t = False
               break
            t = False
      if t : cool_channels.append(i)
         
   return channels,cool_channels

def put_pseudochannel_to_lpm(device,pseudo_channels):  # puts specified pseudochannels to lpm
  for i in pseudo_channels:
   for j in range( i*4, i*4+4 ): device.bankstate[j] = 0
   for j in range( i*4+64, i*4+68 ): device.bankstate[j] = 0
   device.lpm_pseudochannels.add(i)

def put_channel_to_lpm(device,channels):  # puts specified pseudochannels to lpm
  for c in channels:
   i = c*2
   for j in range( i*4, i*4+4 ): device.bankstate[j] = 0
   for j in range( i*4+64, i*4+68 ): device.bankstate[j] = 0
   device.lpm_pseudochannels.add(i)
   i = c*2+1
   for j in range( i*4, i*4+4 ): device.bankstate[j] = 0
   for j in range( i*4+64, i*4+68 ): device.bankstate[j] = 0
   device.lpm_pseudochannels.add(i)
   device.lpm_channels.add(c)
  
def put_pseudochannel_out_lpm(device,pseudo_channels): # puts specified pseudo channels out of lpm
  for i in pseudo_channels:
   for j in range( i*4, i*4+4 ): device.bankstate[j] = 1
   for j in range( i*4+64, i*4+68 ): device.bankstate[j] = 1
   device.lpm_pseudochannels.discard(i)

def put_channel_out_lpm(device,channels):  # puts specified pseudochannels to lpm
  for c in channels:
   i = c*2
   for j in range( i*4, i*4+4 ): device.bankstate[j] = 1
   for j in range( i*4+64, i*4+68 ): device.bankstate[j] = 1
   device.lpm_pseudochannels.discard(i)
   i = c*2+1
   for j in range( i*4, i*4+4 ): device.bankstate[j] = 1
   for j in range( i*4+64, i*4+68 ): device.bankstate[j] = 1
   device.lpm_pseudochannels.discard(i)
   device.lpm_channels.discard(c)

def change_pseudochannel_access_rate(device,channel,change_rd,change_wr):
  for j in range( channel*4, channel*4+4 ):
     device.stats['stat_rd'][j] += change_rd
     if device.stats['stat_rd'][j] < 0: device.stats['stat_rd'][j] = 0
     device.stats['stat_wr'][j] += change_wr
     if device.stats['stat_wr'][j] < 0: device.stats['stat_wr'][j] = 0
  for j in range( channel*4+64, channel*4+68 ):
     device.stats['stat_rd'][j] += change_rd
     if device.stats['stat_rd'][j] < 0: device.stats['stat_rd'][j] = 0
     device.stats['stat_wr'][j] += change_wr
     if device.stats['stat_wr'][j] < 0: device.stats['stat_wr'][j] = 0

def change_channel_access_rate(device,channel,change_rd,change_wr):
  for j in range( channel*8, channel*8+8 ):
     device.stats['stat_rd'][j] += change_rd
     if device.stats['stat_rd'][j] < 0: device.stats['stat_rd'][j] = 0
     device.stats['stat_wr'][j] += change_wr
     if device.stats['stat_wr'][j] < 0: device.stats['stat_wr'][j] = 0
  for j in range( channel*8+64, channel*8+72 ):
     device.stats['stat_rd'][j] += change_rd
     if device.stats['stat_rd'][j] < 0: device.stats['stat_rd'][j] = 0
     device.stats['stat_wr'][j] += change_wr
     if device.stats['stat_wr'][j] < 0: device.stats['stat_wr'][j] = 0

def set_pseudochannel_access_rate(device,channel,change_rd,change_wr):
  for j in range( channel*4, channel*4+4 ):
     device.stats['stat_rd'][j] = change_rd
     device.stats['stat_wr'][j] = change_wr
  for j in range( channel*4+64, channel*4+68 ):
     device.stats['stat_rd'][j] = change_rd
     device.stats['stat_wr'][j] = change_wr

def set_channel_access_rate(device,channel,change_rd,change_wr):
  for j in range( channel*8, channel*8+8 ):
     device.stats['stat_rd'][j] = change_rd
     device.stats['stat_wr'][j] = change_wr
  for j in range( channel*8+64, channel*8+72 ):
     device.stats['stat_rd'][j] = change_rd
     device.stats['stat_wr'][j] = change_wr

def change_bank_access_rate(device,bank,change_rd,change_wr):
  device.stats['stat_rd'][bank] += change_rd
  if device.stats['stat_rd'][bank] < 0: device.stats['stat_rd'][bank] = 0
  device.stats['stat_wr'][bank] += change_wr
  if device.stats['stat_wr'][bank] < 0: device.stats['stat_wr'][bank] = 0

def set_bank_access_rate(device,bank,change_rd,change_wr):
  device.stats['stat_rd'][bank] = change_rd
  device.stats['stat_wr'][bank] = change_wr

# def check_lpm_channel(device,channel):
#    if device.bankstate[channel*4] == 0 : return True
#    else : return False

# def check_lpm_bank(device,bank):
#    if device.bankstate[bank] == 0: return True
#    else : return False

def migrate_channels(device,c1,c2):
   global exec_times, prog_class, max_exec_time_core
   device.migration_activity[c1] = interval_counts_per_migration
   device.migration_activity[c2] = interval_counts_per_migration
   #a = exec_times[2*c1] ; b = exec_times[2*c1+1]
   #exec_times[2*c1] = exec_times[2*c2] ; exec_times[2*c1+1] = exec_times[2*c2+1]
   #exec_times[2*c2] = a ; exec_times[2*c2+1] = b
   cl1 = prog_class[2*c1] ; cl2 = prog_class[2*c1+1]
   prog_class[2*c1] = prog_class[2*c2] ; prog_class[2*c1+1] = prog_class[2*c2+1]
   prog_class[2*c2] = cl1 ; prog_class[2*c2+1] = cl2
   if max_exec_time_core[1]//2 == c1 : 
      if max_exec_time_core[1]%2 == 0: max_exec_time_core[1] = 2*c2
      else:   max_exec_time_core[1] = 2*c2+1
   #print("migrating from channel",c1,"to channel",c2)
   set_channel_access_rate(device,c1,migration_access_per_interval_per_bank,0)
   set_channel_access_rate(device,c2,0,migration_access_per_interval_per_bank)

def figure_out_pseudochannels_neighbours():
   neighbours = []
   for c in range(16):
    neighbour = []
    if c-1>-1 and c%4!=0:  
        neighbour.append(c-1)
    if c-4>-1 : neighbour.append(c-4)
    if c+1<16 and c%4!=3: 
        neighbour.append(c+1)
    if c+4<16 : neighbour.append(c+4)
    neighbours.append(neighbour)
   return neighbours


def figure_out_channels_neighbours():
   neighbours = []
   for c in range(8):
    neighbour = []
    if c-1>-1 and c%2!=0:  
        neighbour.append(c-1)
    if c-2>-1 : neighbour.append(c-2)
    if c+1<8 and c%2!=1: 
        neighbour.append(c+1)
    if c+2<8 : neighbour.append(c+2)
    neighbours.append(neighbour)
   return neighbours


neighbouring_pseudochannels = figure_out_pseudochannels_neighbours()
neighbouring_channels = figure_out_channels_neighbours()

## Thermal Management Policies

# 1 Primitive, put hot channels to lpm until cools down, else sequentially put neighbouring channels to lpm


def dtm1(device):
   #T0,T1,T2,T3,T4 = 74.5,76.4, 76.9, 79.5, 81.0
   #T0_C,T1_C,T2_C,T3_C,T4_C = 72.5,75.4,76.5,77.4, 80.0
   T0,T1,T2,T3,T4 = 76,77, 78, 80, 81.0
   T0_C,T1_C,T2_C,T3_C,T4_C = 74,76.5,77.5,79, 80.5
   # temps is the peak temps of each pseudo-channel
   temps = get_temp_pseudochannels(temperature_trace_file)

   # setting required channels out of lpm
   for c in list(device.lpm_pseudochannels):
      if temps[c] < T0_C:
         put_pseudochannel_out_lpm(device,[c]+neighbouring_pseudochannels[c])
         continue
      elif temps[c] < T1_C:
         put_pseudochannel_out_lpm(device,neighbouring_pseudochannels[c])
         continue
      elif temps[c] < T2_C:
         candidates = [a for a in neighbouring_pseudochannels[c] if a in device.lpm_pseudochannels]
         while len(candidates)>1 :
            mini = np.argmin(temps[candidates])
            put_pseudochannel_out_lpm(device,[candidates[mini]])
            candidates.pop(mini)
         continue
      elif temps[c] < T3_C:
         candidates = [a for a in neighbouring_pseudochannels[c] if a in device.lpm_pseudochannels]
         while len(candidates)>2 :
            mini = np.argmin(temps[candidates])
            put_pseudochannel_out_lpm(device,[candidates[mini]])
            candidates.pop(mini)
         continue
      elif temps[c] < T4_C:
         candidates = [a for a in neighbouring_pseudochannels[c] if a in device.lpm_pseudochannels]
         while len(candidates)>3 :
            mini = np.argmin(temps[candidates])
            put_pseudochannel_out_lpm(device,[candidates[mini]])
            candidates.pop(mini)
         continue
  # now running dtm policy
   temp_order = np.argsort(temps)[::-1]
   i = 0

   while (temps[temp_order[i]]>=T4):
    c = temp_order[i]
    put_pseudochannel_to_lpm(device,[c]+neighbouring_pseudochannels[c])
    i+=1
    if i>15: return

   while (temps[temp_order[i]>=T3]):
    c = temp_order[i]
    lpm_candidates = neighbouring_pseudochannels[c][np.argsort(temps[neighbouring_pseudochannels[c]])[-1:-4:-1]]
    lpm_candidates.append(c)
    put_pseudochannel_to_lpm(device,lpm_candidates)
    i+=1
    if i>15: return
  
   while (temps[temp_order[i]]>=T2):
    c = temp_order[i]
    lpm_candidates = [neighbouring_pseudochannels[c][e] for e in  np.argsort(temps[neighbouring_pseudochannels[c]])[-1:-3:-1] ]
    lpm_candidates.append(c)
    put_pseudochannel_to_lpm(device,lpm_candidates)
    i+=1
    if i>15: return

   while (temps[temp_order[i]]>=T1):
    c = temp_order[i]
    lpm_candidates = [c,neighbouring_pseudochannels[c][np.argmax(temps[neighbouring_pseudochannels[c]])]]
    put_pseudochannel_to_lpm(device,lpm_candidates)
    i+=1
    if i>15: return

   while (temps[temp_order[i]] >= T0):
    c = temp_order[i]
    put_pseudochannel_to_lpm(device,[c])
    i+=1
    if i>15: return

# puts hot channels to lpm until cools down
def dtm2(device):
   T0 = 81.0
   T0_c = 79.0
   temps = get_temp(temperature_trace_file)
   hot,cold = pseudochannels_reached_threshold(temps,T0,T0_c)
   put_pseudochannel_to_lpm(device,hot)
   put_pseudochannel_out_lpm(device,cold)

# same as dtm1 but involving channels level granularity, not pseudochannel level
def dtm3(device): #dtm2 in report
   #T0,T1,T2,T3,T4 = 74.5,76.4, 76.9, 79.5, 81.0
   #T0_C,T1_C,T2_C,T3_C,T4_C = 72.5,75.4,76.5,77.4, 80.0
   T0,T1,T2,T3,T4 = 76,77, 78, 80, 81.0
   T0_C,T1_C,T2_C,T3_C,T4_C = 74,76.5,77.5,79, 80.5
   # temps is the peak temps of each pseudo-channel
   temps = get_temp_channels(temperature_trace_file)

   # setting required channels out of lpm
   for c in list(device.lpm_channels):
      #channel = c//2
      if temps[c] < T0_C:
         put_channel_out_lpm(device,[c]+neighbouring_channels[c])
         continue
      elif temps[c] < T1_C:
         put_channel_out_lpm(device,neighbouring_channels[c])
         continue
      elif temps[c] < T2_C:
         candidates = [a for a in neighbouring_channels[c] if a in device.lpm_channels]
         while len(candidates)>1 :
            mini = np.argmin(temps[candidates])
            put_channel_out_lpm(device,[candidates[mini]])
            candidates.pop(mini)
         continue
      elif temps[c] < T3_C:
         candidates = [a for a in neighbouring_channels[c] if a in device.lpm_channels]
         while len(candidates)>2 :
            mini = np.argmin(temps[candidates])
            put_channel_out_lpm(device,[candidates[mini]])
            candidates.pop(mini)
         continue
      elif temps[c] < T4_C:
         candidates = [a for a in neighbouring_channels[c] if a in device.lpm_channels]
         while len(candidates)>3 :
            mini = np.argmin(temps[candidates])
            put_channel_out_lpm(device,[candidates[mini]])
            candidates.pop(mini)
         continue
  # now running dtm policy
   temp_order = np.argsort(temps)[::-1]
   i = 0

   while (temps[temp_order[i]]>=T4):
    c = temp_order[i]
    put_channel_to_lpm(device,[c]+neighbouring_channels[c])
    i+=1
    if i>7: return

   while (temps[temp_order[i]>=T3]):
    c = temp_order[i]
    lpm_candidates = neighbouring_channels[c][np.argsort(temps[neighbouring_channels[c]])[-1:-4:-1]]
    lpm_candidates.append(c)
    put_channel_to_lpm(device,lpm_candidates)
    i+=1
    if i>7: return
  
   while (temps[temp_order[i]]>=T2):
    c = temp_order[i]
    lpm_candidates = [neighbouring_channels[c][e] for e in  np.argsort(temps[neighbouring_channels[c]])[-1:-3:-1] ]
    lpm_candidates.append(c)
    put_channel_to_lpm(device,lpm_candidates)
    i+=1
    if i>7: return

   while (temps[temp_order[i]]>=T1):
    c = temp_order[i]
    lpm_candidates = [c,neighbouring_channels[c][np.argmax(temps[neighbouring_channels[c]])]]
    put_channel_to_lpm(device,lpm_candidates)
    i+=1
    if i>7: return

   while (temps[temp_order[i]] >= T0):
    c = temp_order[i]
    put_channel_to_lpm(device,[c])
    i+=1
    if i>7: return


# same as dtm2 but involving channels level granularity, not pseudochannel level
def dtm4(device): #dtm1 in report
   T0 = 81.0
   T0_c = 79.0
   temps = get_temp(temperature_trace_file)
   hot,cold = channels_reached_threshold(temps,T0,T0_c)
   put_channel_to_lpm(device,hot)
   put_channel_out_lpm(device,cold)

# same as dtm4, but with layerwise temp thresholds
def dtm6(device): # dtm3 in report
   T0 = [77.5,78,78.5,79,79.5,80,80.5,81]
   T0_c = [75,75.5,76,77,78,78.5,79.5,80.5]
   temps = get_temp(temperature_trace_file)
   hot,cold = channels_reached_threshold_layerwise(temps,T0,T0_c)
   put_channel_to_lpm(device,hot)
   put_channel_out_lpm(device,cold)


# involves both channel swapping and LPM
def figure_out_migration_candidate(device,channel,temps,T0_C):
   # assumptions ; channel is not already migrating

   # heavy = light = mixed = False
   # if device.stats['stat_rd'][channel*8] > 5000 : heavy = True
   # elif device.stats['stat_rd'][channel*8] < 1000 : light = True
   # else: mixed = True
   temp_order = np.argsort(temps)
   i = 1
   if temp_order[0] == channel : return None
   opt = temps[temp_order[0]]
   cand = temp_order[0]
   while (i<8):
      if temp_order[i]==channel or temps[temp_order[i]] - opt > 2.5: break
      if temp_order[i] >= channel+2 : 
         cand = temp_order[i]
         opt = temps[temp_order[i]]
         break
      i+=1
   if opt <= T0_C: return cand
   else: return None

      
def dtm5(device): # dtm4 in report
   #T0,T1,T2,T3,T4 = 76,77, 78, 80, 81.0
   #T0_C,T1_C,T2_C,T3_C,T4_C = 74,76.5,77.5,79, 80.5
   global max_exec_time_core
   T0, T1 = 77, 82
   T0_C, T1_C = 75, 80

   # power is the total power of each channel
   power = get_power_channels(power_trace_file)
   tot_power = get_power_channels(power_trace_file_total)

   static_dynamic_ratio = (tot_power-power)/power
   # temps is the peak temps of each pseudo-channel
   temps = get_temp_channels(temperature_trace_file)

   # setting required channels out of lpm
   for c in list(device.lpm_channels):
      #channel = c//2
      if temps[c] < T1_C:
         put_channel_out_lpm(device,[c])#+neighbouring_channels[c])
         continue

   for c in range(8):
      #if c==max_exec_time_core[1]//2 :
      if static_dynamic_ratio[c] < 2:
         if temps[c] > T0 :
            cand = figure_out_migration_candidate(device,c,temps,T0_C)
            if cand is not None: 
               migrate_channels(device,c,cand)
               with open("migration.txt","a") as f:
                  f.write("Time: %d, Channel: %d, Candidate: %d\n" %(t,c,cand))

      if temps[c] > T1:
         put_channel_to_lpm(device,[c])


## Experiments
np.random.seed(42)

# # 1
# exp = 1
# core_to_pseudochannel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # pseudo-channels allocation to cores
# threshold = 80
# # initial read-write setup
# for i in range(NUM_BANKS):
#   t1.stats['stat_rd'][i] = 0
#   t1.stats['stat_wr'][i] = 0
# # cores 1 and 15 read-write memory accesses increasing linearly
# for t in range(100):
   
#    change_pseudochannel_access_rate(t1,core_to_pseudochannel[1],50*t,40*t)
#    change_pseudochannel_access_rate(t1,core_to_pseudochannel[15],50*t,40*t)

#    t1.calc_temperature_trace(t)
#    # putting channels of cores 1 and 15 to lpm if any channel exceeds threshold
#    temps = get_temp(temperature_trace_file)
#    hot_channels = pseudochannels_reached_threshold(temps,threshold)
#    if len(hot_channels) > 0:
#      print("Hot pseudochannels at time", t, ":", hot_channels)
#      put_pseudochannel_to_lpm(t1,[core_to_pseudochannel[1],core_to_pseudochannel[15]])


# 2 , each core makes random access rate from 0 to 10K to randomly selected banks
# exp = 2
# core_to_pseudochannel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # pseudo-channels allocation to cores
# threshold = 80

# for t in range(100):
#    for core in range(16):
#       r = random.randint(-200,200)
#       change_pseudochannel_access_rate(t1,core_to_pseudochannel[core],r,r)
#       if (random.random()>0.5):
#          r = random.randint(0,5000)
#          bank = pseudochannel_to_banks(core_to_pseudochannel[core])[random.randint(0,7)]
#          change_bank_access_rate(t1,bank,r,r)
#       else:
#           r = random.randint(5000,10000)
#           bank = pseudochannel_to_banks(core_to_pseudochannel[core])[random.randint(0,7)]
#           set_bank_access_rate(t1,bank,r,r)
#    t1.calc_temperature_trace(t)
#    # Putting channels reached threshold to LPM
#    temps = get_temp(temperature_trace_file)
#    hot_channels = pseudochannels_reached_threshold(temps,threshold)
#    put_pseudochannel_to_lpm(t1,hot_channels)






# 3
# exp = 3
# core_to_pseudochannel = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] # pseudo-channels allocation to cores
# threshold = 80

# for i in range(0,16,4):
#    set_pseudochannel_access_rate(t1,i,5000,5000)
# for t in range(300):
#    t1.calc_temperature_trace(t)
#    # Putting channels reached threshold to LPM
#    temps = get_temp(temperature_trace_file)
#    hot_channels = pseudochannels_reached_threshold(temps,threshold)
#    put_pseudochannel_to_lpm(t1,hot_channels)

## MAIN EXPERIMENTS

# 4 memory intensive program on all cores, throughout the sampling interval
# exp = 4
# dtm = 5
# iter = 1000
# stall_count = 0
# stalls = []
# #exec_times = np.random.randint(100,500,16)
# max_exec_time = np.random.randint(100,500)
# max_exec_time_core = [max_exec_time,np.random.randint(0,16)]
# end_time = 0
# prog_class = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# for t in range(iter):
#    for c in range(16):

#       if t1.migration_activity[c] > 0: #pseudochannel is migrating
#          t1.migration_activity[c] -= 1
#          print("migration !!\n")
#          continue

#       #if exec_times[c]>0 and (not(c in t1.lpm_pseudochannels)):
#       if not(c in t1.lpm_pseudochannels): # program is running

#          r = np.random.randint(10000,11000)
#          r = r//2
#          set_pseudochannel_access_rate(t1,c,r,r)

#          # exec_times[c] -= 1
#          # if exec_times[c] == 0:
#          #    end_time = t
#          #    t1.idle_pseudochannels.add(c)

#          if c == max_exec_time_core[1]:
#             max_exec_time_core[0] -= 1

#    t1.calc_temperature_trace(t)
#    # DTM
#    dtm5(t1)
#    # counting stalled cores
#    stall_count += len(t1.lpm_pseudochannels)
#    stalls.append(len(t1.lpm_pseudochannels))
#    # keeping track of max time program is running
#    if max_exec_time_core[0] == 0:
#       end_time = t
#       break
# if end_time == 0:
#    end_time = iter+max_exec_time_core[0]



# 5 memory intensive program on 8 cores, rest programs have mixed pattern
# exp = 5
# dtm = 5
# iter = 1000
# #hotcores = np.random.choice(a=16,size=8,replace=False)
# stall_count = 0
# stalls = []
# #exec_times = np.random.randint(100,500,16)
# end_time = 0
# max_exec_time = np.random.randint(100,500)
# max_exec_time_core = [max_exec_time,np.random.randint(0,16)]
# # heavy = [8,9,10,11,12,13,14,15]
# # mixed = [0,1,2,3,4,5,6,7]
# prog_class = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
# for t in range(iter):
#    for c in range(16):

#       if t1.migration_activity[c] > 0: #pseudochannel is migrating
#          t1.migration_activity[c] -= 1
#          continue

#       #if exec_times[c]>0 and (not(c in t1.lpm_pseudochannels)):
#       if (not(c in t1.lpm_pseudochannels)):  # program is running

#          if prog_class[c] == 1:    # mixed program
#             if (np.random.rand()<0.5):
#                r = np.random.randint(4000,8000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)
#             else:
#                r = np.random.randint(0,4000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)
         
#          else:  # heavy program
#             r = np.random.randint(10000,11000)
#             r = r//2
#             set_pseudochannel_access_rate(t1,c,r,r)

#          # exec_times[c] -= 1
#          # if exec_times[c] == 0:
#          #    end_time = t
#          #    t1.idle_pseudochannels.add(c)

#          if c == max_exec_time_core[1]:
#             max_exec_time_core[0] -= 1

#    t1.calc_temperature_trace(t)
#    #DTM
#    dtm5(t1)
#    # counting stalled cores
#    stall_count += len(t1.lpm_pseudochannels)
#    stalls.append(len(t1.lpm_pseudochannels))
#    # keeping track of max time program is running
#    if max_exec_time_core[0] == 0:
#       end_time = t
#       break
# if end_time == 0:
#    end_time = iter+max_exec_time_core[0]


# 6 memory intensive program on 4 cores, 8 cores with mixed intensity,rest programs have low intensity
# exp = 6
# dtm = 5
# iter = 1000
# # select = np.random.choice(a=16,size=12,replace=False)
# # hotcores = select[:4]
# # mixed = select[4:]

# stall_count = 0
# stalls = []
# #exec_times = np.random.randint(100,500,16)
# end_time = 0
# max_exec_time = np.random.randint(100,500)
# max_exec_time_core = [max_exec_time,np.random.randint(0,16)]
# # heavy = [12,13,14,15]
# # mixed = [4,5,6,7,8,9,10,11]
# # light = [0,1,2,3]
# prog_class = [2,2,2,2,1,1,1,1,1,1,1,1,0,0,0,0]
# for t in range(iter):
#    for c in range(16):

#       if t1.migration_activity[c] > 0: #pseudochannel is migrating
#          t1.migration_activity[c] -= 1
#          continue

#       #if exec_times[c]>0 and (not(c in t1.lpm_pseudochannels)):
#       if (not(c in t1.lpm_pseudochannels)):  # program is running
#          if prog_class[c] == 1: # mixed program
#             if (np.random.rand()<0.5):
#                r = np.random.randint(4000,8000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)
#             else:
#                r = np.random.randint(0,4000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)

#          elif prog_class[c] == 2: # light program
#             if (np.random.rand()<0.02):
#                r = np.random.randint(7000,10000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)
#             else:
#                r = np.random.randint(0,2000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)

#          else: # heavy program
#             r = np.random.randint(10000,11000)
#             r = r//2
#             set_pseudochannel_access_rate(t1,c,r,r)

#          # exec_times[c] -= 1
#          # if exec_times[c] == 0:
#          #    end_time = t
#          #    t1.idle_pseudochannels.add(c)

#          if c == max_exec_time_core[1]:
#             max_exec_time_core[0] -= 1

#    t1.calc_temperature_trace(t)
#    # DTM
#    dtm5(t1)
#    # counting stalled cores
#    stall_count += len(t1.lpm_pseudochannels)
#    stalls.append(len(t1.lpm_pseudochannels))
#    # keeping track of max time program is running
#    if max_exec_time_core[0] == 0:
#       end_time = t
#       break
# if end_time == 0:
#    end_time = iter+max_exec_time_core[0]

# 7 mixed intensity on 4 cores, light on rest
# exp = 7
# dtm = 5
# iter = 1000
# #mixed = np.random.choice(a=16,size=4,replace=False)
# stall_count = 0
# stalls = []
# #exec_times = np.random.randint(100,500,16)
# max_exec_time = np.random.randint(100,500)
# max_exec_time_core = [max_exec_time,np.random.randint(0,16)]
# end_time = 0
# # mixed = [12,13,14,15]
# # light = [0,1,2,3,4,5,6,7,8,9,10,11]
# prog_class = [2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1]
# for t in range(iter):
#    for c in range(16):

#       if t1.migration_activity[c] > 0: #pseudochannel is migrating
#          t1.migration_activity[c] -= 1
#          continue


#       #if exec_times[c]>0 and (not(c in t1.lpm_pseudochannels)):
#       if (not(c in t1.lpm_pseudochannels)):  # program is running
#          if prog_class[2] == 2: # light program
#             if (np.random.rand()<0.02):
#                r = np.random.randint(7000,10000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)
#             else:
#                r = np.random.randint(0,2000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)
#          else: # mixed program
#             if (np.random.rand()<0.5):
#                r = np.random.randint(4000,8000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)
#             else:
#                r = np.random.randint(0,4000)
#                r = r//2
#                set_pseudochannel_access_rate(t1,c,r,r)

#          # exec_times[c] -= 1
#          # if exec_times[c] == 0:
#          #    end_time = t
#          #    t1.idle_pseudochannels.add(c)

#          if c == max_exec_time_core[1]:
#             max_exec_time_core[0] -= 1

#    t1.calc_temperature_trace(t)
#    #DTM
#    dtm5(t1)
#    # counting stalled cores
#    stall_count += len(t1.lpm_pseudochannels)
#    stalls.append(len(t1.lpm_pseudochannels))
#    # keeping track of max time program is running
#    if max_exec_time_core[0] == 0:
#       end_time = t
#       break
# if end_time == 0:
#    end_time = iter+max_exec_time_core[0]


# all light intensity programs
exp = 8
dtm = 5
iter = 1000
stall_count = 0
stalls = []
#exec_times = np.random.randint(100,500,16)
max_exec_time = np.random.randint(100,500)
max_exec_time_core = [max_exec_time,np.random.randint(0,16)]
end_time = 0
prog_class = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
for t in range(iter):
   for c in range(16):

      if t1.migration_activity[c] > 0: #channel(also pseudochannel) is migrating
         t1.migration_activity[c] -= 1
         continue

      #if exec_times[c]>0 and (not(c in t1.lpm_pseudochannels)):
      if  (not(c in t1.lpm_pseudochannels)): # program is running
         if (np.random.rand()<0.05):
            r = np.random.randint(7000,10000)
            r = r//2
            set_pseudochannel_access_rate(t1,c,r,r)
         else:
            r = np.random.randint(0,2000)
            r = r//2
            set_pseudochannel_access_rate(t1,c,r,r)

         # exec_times[c] -= 1
         # if exec_times[c] == 0:
         #    end_time = t
         #    t1.idle_pseudochannels.add(c)

         if c == max_exec_time_core[1]:
            max_exec_time_core[0] -= 1

   t1.calc_temperature_trace(t)
   #DTM
   dtm5(t1)
   # counting stalled cores
   stall_count += len(t1.lpm_pseudochannels)
   stalls.append(len(t1.lpm_pseudochannels))
   # keeping track of max time program is running
   if max_exec_time_core[0] == 0:
      end_time = t
      break
if end_time == 0:
   end_time = iter+max_exec_time_core[0]







# extra

# def dtm2(device):
#    T0 = 81.0
#    temps = get_temp(temperature_trace_file)
#    hot,cold = pseudochannels_reached_threshold(temps,T0)
#    put_pseudochannel_to_lpm(device,hot)
#    put_pseudochannel_out_lpm(device,cold)


# exp = 9
# dtm = 2
# stall_count = 0
# stalls = []
# for t in range(1000):
#    for c in range(16):
#    #   if not(c in t1.lpm_pseudochannels):
#         #r = np.random.randint(10000,11000)
#         #r = r//2
#     r = 1000
#     set_pseudochannel_access_rate(t1,c,r,r)
#    #set_pseudochannel_access_rate(t1,15,r,r)
#    #set_pseudochannel_access_rate(t1,1,r,r)
#    #set_pseudochannel_access_rate(t1,0,r,r)
#    t1.calc_temperature_trace(t)
#       # counting stalled cores
#    stall_count += len(t1.lpm_pseudochannels)
#    stalls.append(len(t1.lpm_pseudochannels))
#    # DTM
#    dtm2(t1)


# exp = 10
# dtm = 0
# end_time = 0
# stall_count = 0
# stalls = []
# for i in range(1000):
#    t1.calc_temperature_trace(i)

# Storing results
os.system(f"mkdir -p ./visualization/{exp}/{dtm}")
os.system(f"cp ./{full_temperature_trace_file} ./visualization/{exp}/{dtm}/{full_temperature_trace_file}")
os.system(f"cp ./{full_power_trace_file} ./visualization/{exp}/{dtm}/{full_power_trace_file}")
os.system(f"cp ./{temperature_trace_file} ./visualization/{exp}/{dtm}/{temperature_trace_file}")
os.system(f"cp ./{power_trace_file} ./visualization/{exp}/{dtm}/{power_trace_file}")
os.system(f"cp ./{hotspot_all_transient_file} ./visualization/{exp}/{dtm}/{hotspot_all_transient_file}")
os.system(f"cp ./{hotspot_steady_temp_file} ./visualization/{exp}/{dtm}/{hotspot_steady_temp_file}")
os.system(f"cp ./{power_trace_file_total} ./visualization/{exp}/{dtm}/{power_trace_file_total}")
os.system(f"cp ./{full_totalpower_trace_file} ./visualization/{exp}/{dtm}/{full_totalpower_trace_file}")

print("Stall count:", stall_count)
print("Stalls:", stalls)
print("Experiment number:", exp)
print("DTM:", dtm)
with open(f"./visualization/{exp}/{dtm}/stalls.txt", "w") as f:
    f.write("Stall count: " + str(stall_count) + "\n")
    f.write("Stalls: "+"\n")
    for i in stalls:
        f.write(str(i) + "\n")
    print("Stalls written to file!")
with open(f"./visualization/{exp}/{dtm}/endtime.txt", "w") as f:
    f.write("End time: "+ str(end_time) + "( "+str(end_time*interval_sec)+ " s )\n")
    print(f"Endtime {end_time} ( {end_time*interval_sec} s ) written to file!")