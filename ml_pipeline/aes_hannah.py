import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from PIL import Image 
import numpy as np

def add_binaries(m1, m2):
    r1, c1 = m1.shape
    r2, c2 = m2.shape

    if r1 != r2 or c1 != c2:
        return None
    
    res = []
    for i in range(r1):
        row = []
        for j in range(c1):
            bin1 = int(np.binary_repr(m1[i,j]))
            bin2 = int(np.binary_repr(m2[i,j]))
            row.append(int(np.base_repr(bin1 ^ bin2, 10)))
        res.append(row)
    return np.array(res)

def simple_add(ct, sacrifice, r, c, noise_ratio):
    num_pixels = r*c
    r_flat = np.frombuffer(ct[:num_pixels], dtype=np.uint8)
    g_flat = np.frombuffer(ct[num_pixels: num_pixels*2], dtype=np.uint8)
    b_flat = np.frombuffer(ct[num_pixels*2:], dtype=np.uint8)

    offset_r = np.reshape(r_flat, (r, c))
    offset_g = np.reshape(g_flat, (r, c))
    offset_b = np.reshape(b_flat, (r, c))

    noise = np.stack((offset_r, offset_g, offset_b), axis=2)

    new_major = Image.fromarray(np.add(noise//noise_ratio, sacrifice))
    new_major.show()

def pixel_xor(ct, sacrifice, r, c, noise_ratio):
    num_pixels = r*c
    r_flat = np.frombuffer(ct[:num_pixels], dtype=np.uint8)
    g_flat = np.frombuffer(ct[num_pixels: num_pixels*2], dtype=np.uint8)
    b_flat = np.frombuffer(ct[num_pixels*2:], dtype=np.uint8)

    offset_r = np.reshape(r_flat//noise_ratio, (r, c))
    offset_g = np.reshape(g_flat//noise_ratio, (r, c))
    offset_b = np.reshape(b_flat//noise_ratio, (r, c))

    noised_red = add_binaries(offset_r, sacrifice[:,:,0])
    noised_green = add_binaries(offset_g, sacrifice[:,:,1])
    noised_blue = add_binaries(offset_b, sacrifice[:,:,2])

    noised_major = np.stack((noised_red, noised_green, noised_blue), axis=2)

    new_major = Image.fromarray(noised_major.astype(np.uint8))
    new_major.show() 

def red_xor(ct, sacrifice, r, c, noise_ratio):
    num_pixels = r*c
    r_flat = np.frombuffer(ct[:num_pixels], dtype=np.uint8)

    offset_r = np.reshape(r_flat//noise_ratio, (r, c))

    noised_red = add_binaries(offset_r, sacrifice[:,:,0])

    noised_major = np.stack((noised_red, sacrifice[:,:,1], sacrifice[:,:,2]), axis=2)

    new_major = Image.fromarray(noised_major.astype(np.uint8))
    new_major.show() 

def green_xor(ct, sacrifice, r, c, noise_ratio):
    num_pixels = r*c
    g_flat = np.frombuffer(ct[num_pixels: num_pixels*2], dtype=np.uint8)

    offset_g = np.reshape(g_flat//noise_ratio, (r, c))

    noised_green = add_binaries(offset_g, sacrifice[:,:,0])

    noised_major = np.stack((sacrifice[:,:,0], noised_green, sacrifice[:,:,2]), axis=2)

    new_major = Image.fromarray(noised_major.astype(np.uint8))
    new_major.show() 

def blue_xor(ct, sacrifice, r, c, noise_ratio):
    num_pixels = r*c
    b_flat = np.frombuffer(ct[num_pixels*2:], dtype=np.uint8)

    offset_b = np.reshape(b_flat//noise_ratio, (r, c))

    noised_blue = add_binaries(offset_b, sacrifice[:,:,0])

    noised_major = np.stack((sacrifice[:,:,0], sacrifice[:,:,1], noised_blue), axis=2)

    new_major = Image.fromarray(noised_major.astype(np.uint8))
    new_major.show() 

def main():
    key = os.urandom(32) # we should change this later to be artist dependent?
    init_vector = os.urandom(16) # must be 16*8 = 128 bits

    # in order for encrypt to work this thing has to be a multiple of 16 long!!!
    # who would have thunk
    sacrifice = np.array(Image.open("./major_sacrifice.jpg"))
    r, c, _ = sacrifice.shape
    num_pixels = r*c

    # mode is currently CBC -- we can play around with this
    cipher = Cipher(algorithms.AES(key), modes.CBC(init_vector))
    encryptor = cipher.encryptor()
    decryptor = cipher.decryptor()

    # time to f around
    # i just want to apply this stupid AES to each channel in a pixel
    ct = b""
    pad = b""
    if num_pixels % 16 != 0:
        tc = num_pixels//16 * 16
        pad = b"0" * tc

    for i in range(3):
        pic_channel = sacrifice[:,:,i].tobytes()
        ct += encryptor.update(pic_channel+pad)
    ct += encryptor.finalize()

    # simple_add(ct, sacrifice, r, c)
    # pixel_xor(ct, sacrifice, r, c, 100)
    red_xor(ct, sacrifice, r, c, 50)
    green_xor(ct, sacrifice, r, c, 50)
    blue_xor(ct, sacrifice, r, c, 50)

if __name__ == "__main__":
    main()