import const.indices as indices

class Constants:

    headers = ['0_mean', '1_mean', '2_mean', '3_mean', '4_mean', '5_mean', '6_mean', '7_mean', '8_mean', '9_mean', '10_mean', '11_mean', '12_mean', '13_mean', '14_mean', '15_mean', '16_mean', '17_mean', '18_mean', '19_mean', '20_mean', '21_mean', '22_mean', '23_mean', '24_mean', '25_mean', '26_mean', '27_mean', '28_mean', '29_mean', '30_mean', '31_mean', '32_mean', '33_mean', '34_mean', '35_mean', '36_mean', '37_mean', '38_mean', '39_mean', '40_mean', '41_mean', '42_mean', '43_mean', '44_mean', '45_mean', '46_mean', '47_mean', '48_mean', '49_mean', '50_mean', '51_mean', '52_mean', '53_mean', '54_mean', '55_mean', '56_mean', '57_mean', '58_mean', '59_mean', '60_mean', '61_mean', '62_mean', '63_mean', '64_mean', '65_mean', '0_std', '1_std', '2_std', '3_std', '4_std', '5_std', '6_std', '7_std', '8_std', '9_std', '10_std', '11_std', '12_std', '13_std', '14_std', '15_std', '16_std', '17_std', '18_std', '19_std', '20_std', '21_std', '22_std', '23_std', '24_std', '25_std', '26_std', '27_std', '28_std', '29_std', '30_std', '31_std', '32_std', '33_std', '34_std', '35_std', '36_std', '37_std', '38_std', '39_std', '40_std', '41_std', '42_std', '43_std', '44_std', '45_std', '46_std', '47_std', '48_std', '49_std', '50_std', '51_std', '52_std', '53_std', '54_std', '55_std', '56_std', '57_std', '58_std', '59_std', '60_std', '61_std', '62_std', '63_std', '64_std', '65_std', '0_min', '1_min', '2_min', '3_min', '4_min', '5_min', '6_min', '7_min', '8_min', '9_min', '10_min', '11_min', '12_min', '13_min', '14_min', '15_min', '16_min', '17_min', '18_min', '19_min', '20_min', '21_min', '22_min', '23_min', '24_min', '25_min', '26_min', '27_min', '28_min', '29_min', '30_min', '31_min', '32_min', '33_min', '34_min', '35_min', '36_min', '37_min', '38_min', '39_min', '40_min', '41_min', '42_min', '43_min', '44_min', '45_min', '46_min', '47_min', '48_min', '49_min', '50_min', '51_min', '52_min', '53_min', '54_min', '55_min', '56_min', '57_min', '58_min', '59_min', '60_min', '61_min', '62_min', '63_min', '64_min', '65_min', '0_25q', '1_25q', '2_25q', '3_25q', '4_25q', '5_25q', '6_25q', '7_25q', '8_25q', '9_25q', '10_25q', '11_25q', '12_25q', '13_25q', '14_25q', '15_25q', '16_25q', '17_25q', '18_25q', '19_25q', '20_25q', '21_25q', '22_25q', '23_25q', '24_25q', '25_25q', '26_25q', '27_25q', '28_25q', '29_25q', '30_25q', '31_25q', '32_25q', '33_25q', '34_25q', '35_25q', '36_25q', '37_25q', '38_25q', '39_25q', '40_25q', '41_25q', '42_25q', '43_25q', '44_25q', '45_25q', '46_25q', '47_25q', '48_25q', '49_25q', '50_25q', '51_25q', '52_25q', '53_25q', '54_25q', '55_25q', '56_25q', '57_25q', '58_25q', '59_25q', '60_25q', '61_25q', '62_25q', '63_25q', '64_25q', '65_25q', '0_median', '1_median', '2_median', '3_median', '4_median', '5_median', '6_median', '7_median', '8_median', '9_median', '10_median', '11_median', '12_median', '13_median', '14_median', '15_median', '16_median', '17_median', '18_median', '19_median', '20_median', '21_median', '22_median', '23_median', '24_median', '25_median', '26_median', '27_median', '28_median', '29_median', '30_median', '31_median', '32_median', '33_median', '34_median', '35_median', '36_median', '37_median', '38_median', '39_median', '40_median', '41_median', '42_median', '43_median', '44_median', '45_median', '46_median', '47_median', '48_median', '49_median', '50_median', '51_median', '52_median', '53_median', '54_median', '55_median', '56_median', '57_median', '58_median', '59_median', '60_median', '61_median', '62_median', '63_median', '64_median', '65_median', '0_75q', '1_75q', '2_75q', '3_75q', '4_75q', '5_75q', '6_75q', '7_75q', '8_75q', '9_75q', '10_75q', '11_75q', '12_75q', '13_75q', '14_75q', '15_75q', '16_75q', '17_75q', '18_75q', '19_75q', '20_75q', '21_75q', '22_75q', '23_75q', '24_75q', '25_75q', '26_75q', '27_75q', '28_75q', '29_75q', '30_75q', '31_75q', '32_75q', '33_75q', '34_75q', '35_75q', '36_75q', '37_75q', '38_75q', '39_75q', '40_75q', '41_75q', '42_75q', '43_75q', '44_75q', '45_75q', '46_75q', '47_75q', '48_75q', '49_75q', '50_75q', '51_75q', '52_75q', '53_75q', '54_75q', '55_75q', '56_75q', '57_75q', '58_75q', '59_75q', '60_75q', '61_75q', '62_75q', '63_75q', '64_75q', '65_75q', '0_max', '1_max', '2_max', '3_max', '4_max', '5_max', '6_max', '7_max', '8_max', '9_max', '10_max', '11_max', '12_max', '13_max', '14_max', '15_max', '16_max', '17_max', '18_max', '19_max', '20_max', '21_max', '22_max', '23_max', '24_max', '25_max', '26_max', '27_max', '28_max', '29_max', '30_max', '31_max', '32_max', '33_max', '34_max', '35_max', '36_max', '37_max', '38_max', '39_max', '40_max', '41_max', '42_max', '43_max', '44_max', '45_max', '46_max', '47_max', '48_max', '49_max', '50_max', '51_max', '52_max', '53_max', '54_max', '55_max', '56_max', '57_max', '58_max', '59_max', '60_max', '61_max', '62_max', '63_max', '64_max', '65_max', '0_min_max', '1_min_max', '2_min_max', '3_min_max', '4_min_max', '5_min_max', '6_min_max', '7_min_max', '8_min_max', '9_min_max', '10_min_max', '11_min_max', '12_min_max', '13_min_max', '14_min_max', '15_min_max', '16_min_max', '17_min_max', '18_min_max', '19_min_max', '20_min_max', '21_min_max', '22_min_max', '23_min_max', '24_min_max', '25_min_max', '26_min_max', '27_min_max', '28_min_max', '29_min_max', '30_min_max', '31_min_max', '32_min_max', '33_min_max', '34_min_max', '35_min_max', '36_min_max', '37_min_max', '38_min_max', '39_min_max', '40_min_max', '41_min_max', '42_min_max', '43_min_max', '44_min_max', '45_min_max', '46_min_max', '47_min_max', '48_min_max', '49_min_max', '50_min_max', '51_min_max', '52_min_max', '53_min_max', '54_min_max', '55_min_max', '56_min_max', '57_min_max', '58_min_max', '59_min_max', '60_min_max', '61_min_max', '62_min_max', '63_min_max', '64_min_max', '65_min_max', '0_var', '1_var', '2_var', '3_var', '4_var', '5_var', '6_var', '7_var', '8_var', '9_var', '10_var', '11_var', '12_var', '13_var', '14_var', '15_var', '16_var', '17_var', '18_var', '19_var', '20_var', '21_var', '22_var', '23_var', '24_var', '25_var', '26_var', '27_var', '28_var', '29_var', '30_var', '31_var', '32_var', '33_var', '34_var', '35_var', '36_var', '37_var', '38_var', '39_var', '40_var', '41_var', '42_var', '43_var', '44_var', '45_var', '46_var', '47_var', '48_var', '49_var', '50_var', '51_var', '52_var', '53_var', '54_var', '55_var', '56_var', '57_var', '58_var', '59_var', '60_var', '61_var', '62_var', '63_var', '64_var', '65_var', '0_skew', '1_skew', '2_skew', '3_skew', '4_skew', '5_skew', '6_skew', '7_skew', '8_skew', '9_skew', '10_skew', '11_skew', '12_skew', '13_skew', '14_skew', '15_skew', '16_skew', '17_skew', '18_skew', '19_skew', '20_skew', '21_skew', '22_skew', '23_skew', '24_skew', '25_skew', '26_skew', '27_skew', '28_skew', '29_skew', '30_skew', '31_skew', '32_skew', '33_skew', '34_skew', '35_skew', '36_skew', '37_skew', '38_skew', '39_skew', '40_skew', '41_skew', '42_skew', '43_skew', '44_skew', '45_skew', '46_skew', '47_skew', '48_skew', '49_skew', '50_skew', '51_skew', '52_skew', '53_skew', '54_skew', '55_skew', '56_skew', '57_skew', '58_skew', '59_skew', '60_skew', '61_skew', '62_skew', '63_skew', '64_skew', '65_skew', '0_kurtosis', '1_kurtosis', '2_kurtosis', '3_kurtosis', '4_kurtosis', '5_kurtosis', '6_kurtosis', '7_kurtosis', '8_kurtosis', '9_kurtosis', '10_kurtosis', '11_kurtosis', '12_kurtosis', '13_kurtosis', '14_kurtosis', '15_kurtosis', '16_kurtosis', '17_kurtosis', '18_kurtosis', '19_kurtosis', '20_kurtosis', '21_kurtosis', '22_kurtosis', '23_kurtosis', '24_kurtosis', '25_kurtosis', '26_kurtosis', '27_kurtosis', '28_kurtosis', '29_kurtosis', '30_kurtosis', '31_kurtosis', '32_kurtosis', '33_kurtosis', '34_kurtosis', '35_kurtosis', '36_kurtosis', '37_kurtosis', '38_kurtosis', '39_kurtosis', '40_kurtosis', '41_kurtosis', '42_kurtosis', '43_kurtosis', '44_kurtosis', '45_kurtosis', '46_kurtosis', '47_kurtosis', '48_kurtosis', '49_kurtosis', '50_kurtosis', '51_kurtosis', '52_kurtosis', '53_kurtosis', '54_kurtosis', '55_kurtosis', '56_kurtosis', '57_kurtosis', '58_kurtosis', '59_kurtosis', '60_kurtosis', '61_kurtosis', '62_kurtosis', '63_kurtosis', '64_kurtosis', '65_kurtosis', '0_mode', '1_mode', '2_mode', '3_mode', '4_mode', '5_mode', '6_mode', '7_mode', '8_mode', '9_mode', '10_mode', '11_mode', '12_mode', '13_mode', '14_mode', '15_mode', '16_mode', '17_mode', '18_mode', '19_mode', '20_mode', '21_mode', '22_mode', '23_mode', '24_mode', '25_mode', '26_mode', '27_mode', '28_mode', '29_mode', '30_mode', '31_mode', '32_mode', '33_mode', '34_mode', '35_mode', '36_mode', '37_mode', '38_mode', '39_mode', '40_mode', '41_mode', '42_mode', '43_mode', '44_mode', '45_mode', '46_mode', '47_mode', '48_mode', '49_mode', '50_mode', '51_mode', '52_mode', '53_mode', '54_mode', '55_mode', '56_mode', '57_mode', '58_mode', '59_mode', '60_mode', '61_mode', '62_mode', '63_mode', '64_mode', '65_mode', '0_spectral_centeroid', '1_spectral_centeroid', '2_spectral_centeroid', '3_spectral_centeroid', '4_spectral_centeroid', '5_spectral_centeroid', '6_spectral_centeroid', '7_spectral_centeroid', '8_spectral_centeroid', '9_spectral_centeroid', '10_spectral_centeroid', '11_spectral_centeroid', '12_spectral_centeroid', '13_spectral_centeroid', '14_spectral_centeroid', '15_spectral_centeroid', '16_spectral_centeroid', '17_spectral_centeroid', '18_spectral_centeroid', '19_spectral_centeroid', '20_spectral_centeroid', '21_spectral_centeroid', '22_spectral_centeroid', '23_spectral_centeroid', '24_spectral_centeroid', '25_spectral_centeroid', '26_spectral_centeroid', '27_spectral_centeroid', '28_spectral_centeroid', '29_spectral_centeroid', '30_spectral_centeroid', '31_spectral_centeroid', '32_spectral_centeroid', '33_spectral_centeroid', '34_spectral_centeroid', '35_spectral_centeroid', '36_spectral_centeroid', '37_spectral_centeroid', '38_spectral_centeroid', '39_spectral_centeroid', '40_spectral_centeroid', '41_spectral_centeroid', '42_spectral_centeroid', '43_spectral_centeroid', '44_spectral_centeroid', '45_spectral_centeroid', '46_spectral_centeroid', '47_spectral_centeroid', '48_spectral_centeroid', '49_spectral_centeroid', '50_spectral_centeroid', '51_spectral_centeroid', '52_spectral_centeroid', '53_spectral_centeroid', '54_spectral_centeroid', '55_spectral_centeroid', '56_spectral_centeroid', '57_spectral_centeroid', '58_spectral_centeroid', '59_spectral_centeroid', '60_spectral_centeroid', '61_spectral_centeroid', '62_spectral_centeroid', '63_spectral_centeroid', '64_spectral_centeroid', '65_spectral_centeroid', '0_spectral_entropy', '1_spectral_entropy', '2_spectral_entropy', '3_spectral_entropy', '4_spectral_entropy', '5_spectral_entropy', '6_spectral_entropy', '7_spectral_entropy', '8_spectral_entropy', '9_spectral_entropy', '10_spectral_entropy', '11_spectral_entropy', '12_spectral_entropy', '13_spectral_entropy', '14_spectral_entropy', '15_spectral_entropy', '16_spectral_entropy', '17_spectral_entropy', '18_spectral_entropy', '19_spectral_entropy', '20_spectral_entropy', '21_spectral_entropy', '22_spectral_entropy', '23_spectral_entropy', '24_spectral_entropy', '25_spectral_entropy', '26_spectral_entropy', '27_spectral_entropy', '28_spectral_entropy', '29_spectral_entropy', '30_spectral_entropy', '31_spectral_entropy', '32_spectral_entropy', '33_spectral_entropy', '34_spectral_entropy', '35_spectral_entropy', '36_spectral_entropy', '37_spectral_entropy', '38_spectral_entropy', '39_spectral_entropy', '40_spectral_entropy', '41_spectral_entropy', '42_spectral_entropy', '43_spectral_entropy', '44_spectral_entropy', '45_spectral_entropy', '46_spectral_entropy', '47_spectral_entropy', '48_spectral_entropy', '49_spectral_entropy', '50_spectral_entropy', '51_spectral_entropy', '52_spectral_entropy', '53_spectral_entropy', '54_spectral_entropy', '55_spectral_entropy', '56_spectral_entropy', '57_spectral_entropy', '58_spectral_entropy', '59_spectral_entropy', '60_spectral_entropy', '61_spectral_entropy', '62_spectral_entropy', '63_spectral_entropy', '64_spectral_entropy', '65_spectral_entropy', '0_ff1', '1_ff1', '2_ff1', '3_ff1', '4_ff1', '5_ff1', '6_ff1', '7_ff1', '8_ff1', '9_ff1', '10_ff1', '11_ff1', '12_ff1', '13_ff1', '14_ff1', '15_ff1', '16_ff1', '17_ff1', '18_ff1', '19_ff1', '20_ff1', '21_ff1', '22_ff1', '23_ff1', '24_ff1', '25_ff1', '26_ff1', '27_ff1', '28_ff1', '29_ff1', '30_ff1', '31_ff1', '32_ff1', '33_ff1', '34_ff1', '35_ff1', '36_ff1', '37_ff1', '38_ff1', '39_ff1', '40_ff1', '41_ff1', '42_ff1', '43_ff1', '44_ff1', '45_ff1', '46_ff1', '47_ff1', '48_ff1', '49_ff1', '50_ff1', '51_ff1', '52_ff1', '53_ff1', '54_ff1', '55_ff1', '56_ff1', '57_ff1', '58_ff1', '59_ff1', '60_ff1', '61_ff1', '62_ff1', '63_ff1', '64_ff1', '65_ff1', '0_ff2', '1_ff2', '2_ff2', '3_ff2', '4_ff2', '5_ff2', '6_ff2', '7_ff2', '8_ff2', '9_ff2', '10_ff2', '11_ff2', '12_ff2', '13_ff2', '14_ff2', '15_ff2', '16_ff2', '17_ff2', '18_ff2', '19_ff2', '20_ff2', '21_ff2', '22_ff2', '23_ff2', '24_ff2', '25_ff2', '26_ff2', '27_ff2', '28_ff2', '29_ff2', '30_ff2', '31_ff2', '32_ff2', '33_ff2', '34_ff2', '35_ff2', '36_ff2', '37_ff2', '38_ff2', '39_ff2', '40_ff2', '41_ff2', '42_ff2', '43_ff2', '44_ff2', '45_ff2', '46_ff2', '47_ff2', '48_ff2', '49_ff2', '50_ff2', '51_ff2', '52_ff2', '53_ff2', '54_ff2', '55_ff2', '56_ff2', '57_ff2', '58_ff2', '59_ff2', '60_ff2', '61_ff2', '62_ff2', '63_ff2', '64_ff2', '65_ff2', '0_ff3', '1_ff3', '2_ff3', '3_ff3', '4_ff3', '5_ff3', '6_ff3', '7_ff3', '8_ff3', '9_ff3', '10_ff3', '11_ff3', '12_ff3', '13_ff3', '14_ff3', '15_ff3', '16_ff3', '17_ff3', '18_ff3', '19_ff3', '20_ff3', '21_ff3', '22_ff3', '23_ff3', '24_ff3', '25_ff3', '26_ff3', '27_ff3', '28_ff3', '29_ff3', '30_ff3', '31_ff3', '32_ff3', '33_ff3', '34_ff3', '35_ff3', '36_ff3', '37_ff3', '38_ff3', '39_ff3', '40_ff3', '41_ff3', '42_ff3', '43_ff3', '44_ff3', '45_ff3', '46_ff3', '47_ff3', '48_ff3', '49_ff3', '50_ff3', '51_ff3', '52_ff3', '53_ff3', '54_ff3', '55_ff3', '56_ff3', '57_ff3', '58_ff3', '59_ff3', '60_ff3', '61_ff3', '62_ff3', '63_ff3', '64_ff3', '65_ff3', '0_ff4', '1_ff4', '2_ff4', '3_ff4', '4_ff4', '5_ff4', '6_ff4', '7_ff4', '8_ff4', '9_ff4', '10_ff4', '11_ff4', '12_ff4', '13_ff4', '14_ff4', '15_ff4', '16_ff4', '17_ff4', '18_ff4', '19_ff4', '20_ff4', '21_ff4', '22_ff4', '23_ff4', '24_ff4', '25_ff4', '26_ff4', '27_ff4', '28_ff4', '29_ff4', '30_ff4', '31_ff4', '32_ff4', '33_ff4', '34_ff4', '35_ff4', '36_ff4', '37_ff4', '38_ff4', '39_ff4', '40_ff4', '41_ff4', '42_ff4', '43_ff4', '44_ff4', '45_ff4', '46_ff4', '47_ff4', '48_ff4', '49_ff4', '50_ff4', '51_ff4', '52_ff4', '53_ff4', '54_ff4', '55_ff4', '56_ff4', '57_ff4', '58_ff4', '59_ff4', '60_ff4', '61_ff4', '62_ff4', '63_ff4', '64_ff4', '65_ff4', '0_ff5', '1_ff5', '2_ff5', '3_ff5', '4_ff5', '5_ff5', '6_ff5', '7_ff5', '8_ff5', '9_ff5', '10_ff5', '11_ff5', '12_ff5', '13_ff5', '14_ff5', '15_ff5', '16_ff5', '17_ff5', '18_ff5', '19_ff5', '20_ff5', '21_ff5', '22_ff5', '23_ff5', '24_ff5', '25_ff5', '26_ff5', '27_ff5', '28_ff5', '29_ff5', '30_ff5', '31_ff5', '32_ff5', '33_ff5', '34_ff5', '35_ff5', '36_ff5', '37_ff5', '38_ff5', '39_ff5', '40_ff5', '41_ff5', '42_ff5', '43_ff5', '44_ff5', '45_ff5', '46_ff5', '47_ff5', '48_ff5', '49_ff5', '50_ff5', '51_ff5', '52_ff5', '53_ff5', '54_ff5', '55_ff5', '56_ff5', '57_ff5', '58_ff5', '59_ff5', '60_ff5', '61_ff5', '62_ff5', '63_ff5', '64_ff5', '65_ff5', '0_freq_5sum', '1_freq_5sum', '2_freq_5sum', '3_freq_5sum', '4_freq_5sum', '5_freq_5sum', '6_freq_5sum', '7_freq_5sum', '8_freq_5sum', '9_freq_5sum', '10_freq_5sum', '11_freq_5sum', '12_freq_5sum', '13_freq_5sum', '14_freq_5sum', '15_freq_5sum', '16_freq_5sum', '17_freq_5sum', '18_freq_5sum', '19_freq_5sum', '20_freq_5sum', '21_freq_5sum', '22_freq_5sum', '23_freq_5sum', '24_freq_5sum', '25_freq_5sum', '26_freq_5sum', '27_freq_5sum', '28_freq_5sum', '29_freq_5sum', '30_freq_5sum', '31_freq_5sum', '32_freq_5sum', '33_freq_5sum', '34_freq_5sum', '35_freq_5sum', '36_freq_5sum', '37_freq_5sum', '38_freq_5sum', '39_freq_5sum', '40_freq_5sum', '41_freq_5sum', '42_freq_5sum', '43_freq_5sum', '44_freq_5sum', '45_freq_5sum', '46_freq_5sum', '47_freq_5sum', '48_freq_5sum', '49_freq_5sum', '50_freq_5sum', '51_freq_5sum', '52_freq_5sum', '53_freq_5sum', '54_freq_5sum', '55_freq_5sum', '56_freq_5sum', '57_freq_5sum', '58_freq_5sum', '59_freq_5sum', '60_freq_5sum', '61_freq_5sum', '62_freq_5sum', '63_freq_5sum', '64_freq_5sum', '65_freq_5sum', '0_bandwidth', '1_bandwidth', '2_bandwidth', '3_bandwidth', '4_bandwidth', '5_bandwidth', '6_bandwidth', '7_bandwidth', '8_bandwidth', '9_bandwidth', '10_bandwidth', '11_bandwidth', '12_bandwidth', '13_bandwidth', '14_bandwidth', '15_bandwidth', '16_bandwidth', '17_bandwidth', '18_bandwidth', '19_bandwidth', '20_bandwidth', '21_bandwidth', '22_bandwidth', '23_bandwidth', '24_bandwidth', '25_bandwidth', '26_bandwidth', '27_bandwidth', '28_bandwidth', '29_bandwidth', '30_bandwidth', '31_bandwidth', '32_bandwidth', '33_bandwidth', '34_bandwidth', '35_bandwidth', '36_bandwidth', '37_bandwidth', '38_bandwidth', '39_bandwidth', '40_bandwidth', '41_bandwidth', '42_bandwidth', '43_bandwidth', '44_bandwidth', '45_bandwidth', '46_bandwidth', '47_bandwidth', '48_bandwidth', '49_bandwidth', '50_bandwidth', '51_bandwidth', '52_bandwidth', '53_bandwidth', '54_bandwidth', '55_bandwidth', '56_bandwidth', '57_bandwidth', '58_bandwidth', '59_bandwidth', '60_bandwidth', '61_bandwidth', '62_bandwidth', '63_bandwidth', '64_bandwidth', '65_bandwidth']

    gesture_names = ['gesture 1']

    window_size = 90
    window_distance = 30

    flex_map = [indices.thumb_base,
                indices.thumb_tip,
                indices.thumb_angle,
                indices.finger1_base,
                indices.finger1_tip,
                indices.finger2_base,
                indices.finger2_tip,
                indices.finger3_base,
                indices.finger3_tip,
                indices.finger4_base,
                indices.finger4_tip,
                indices.wrist_flexion,
                indices.wrist_extension]

    hand_row_1 = [indices.thumb_base,
                indices.finger1_base,
                indices.finger2_base,
                indices.finger3_base,
                indices.finger4_base]

    hand_row_2 = [indices.thumb_tip,
                indices.finger1_tip,
                indices.finger2_tip,
                indices.finger3_tip,
                indices.finger4_tip]