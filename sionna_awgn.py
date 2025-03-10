import tensorflow as tf
import sionna
from sionna.utils import BinarySource, compute_ser, compute_ber
from sionna.channel import AWGN
from sionna.mapping import SymbolDemapper, Mapper, Demapper, Constellation

# Define channel parameters
num_tx_ant = 4 
num_rx_ant = 4
num_bits_per_symbol = 2
batch_size = 100
codeword_length = 1024  # bits
SNR_dB = 20.  # Transmit SNR [dB]

sionna.config.seed = 42  # Reproducibility

constellation = Constellation("qam", num_bits_per_symbol)
constellation.show()

binary_source = BinarySource()
b = binary_source([batch_size, num_tx_ant, codeword_length])
mapper = Mapper(constellation=constellation)
x = mapper(b)

# Find the average energy of the transmitted signal x
average_signal_energy = tf.reduce_mean(tf.abs(x) ** 2)
noise_power = average_signal_energy / 10 ** (SNR_dB / 10.)

# Build an AWGN channel
channel = AWGN()
y = channel([x, noise_power])
x_hat = tf.identity(y)  # No equalization here.

# Symbol detection
symbol_demapper = SymbolDemapper('qam', num_bits_per_symbol, hard_out=True)
x_ind = symbol_demapper([x, noise_power])
x_hat_ind = symbol_demapper([x_hat, noise_power])
ser = compute_ser(x_ind, x_hat_ind)

# Convert symbols to bits and compute BER.
demapper = Demapper('app', constellation=constellation) 
llr = demapper([y, noise_power])
b_hat = tf.cast(tf.less(0., llr), dtype=tf.float32)

ber = compute_ber(b, b_hat)

print(SNR_dB, ser.numpy(), ber.numpy())
