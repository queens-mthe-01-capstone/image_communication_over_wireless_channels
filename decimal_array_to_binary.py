import numpy as np
import math

def decimal_array_to_binary(numbers_array, bits_per_bin_num_array):
#inputs are array of values (1D), array of bits per value (1D) and same size
#assumption is that if the bits entry is not 0, then the corresponding value can be represented by the given number of bits
#assumption is that once a 0 is hit in the bits per value array, the rest will be 0s too, so it will break out of the loop
#output is the np array of the binary numbers of the values, with number of bits as given in the bits array

  numbers_array = np.asarray(numbers_array)
  bits_per_bin_num_array = np.asarray(bits_per_bin_num_array)
  ret_list = []
  for i, element in np.ndenumerate(numbers_array):
    bits_per_bin_num = bits_per_bin_num_array[i]
    if bits_per_bin_num == 0:
      break
    num_digits = math.floor(math.log2(element)) + 1
    if num_digits > bits_per_bin_num:
      raise ValueError("Need more bits per number for the given values")
    diff = bits_per_bin_num - num_digits
    ret_list.extend([0] * diff)
    ret_list.extend([int(bit) for bit in bin(element)[2:]])
  return np.array(ret_list, dtype=np.uint8)