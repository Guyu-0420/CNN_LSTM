# CNN_LSTM
Prediction of Intraoperative hypotension

data1（代码使用） : 18个病人的术中数据（部分） 
链接：https://pan.baidu.com/s/1QSt1qPEeYkYYu5K3xZISeQ?pwd=ru8i 
提取码：ru8i

data2：未预处理的全部数据
链接：https://pan.baidu.com/s/1jgEAEnpTcT94cNmR-jVoHA?pwd=l8vn 
提取码：l8vn

项目内容：
1.使用 18 个患者的 10 个与 MAP 强相关的特征进行训练。
2.模型选择：由于数据和时间序列强相关，因此我们选用了 CNN 与 LSTM 相结合的方式，有效的提高了学习效率，同时保证了模型拟合质量。
3.实验目的：
  基于大量术中生理数据样本，使用深度学习算法，对正常生理状态下 MAP进行预测，通过与术中实测值相比对，可以确定患者血压异常时刻，对患者术后复盘和后续治疗方案的确定起到一定作用

主要贡献：
1.1-5号病人为测试集，6-18号病人为训练集，训练集进行12折交叉验证
2。使用CNN+LSTM
3.使用滑动时间窗，增加数据量

可改进的工作：
1.原始数据的预处理 未做
2.用PCA对原始数据进行降维 未做

全部数据内容及定义：
* 定义低血压事件为手术期间患者 MAP 维持在小于 65mmHg 范围并持续 1 分钟
* 数据集中包含 18 个病人在不同时刻通过四种仪器采集的共 57 个特征，其值随时间连续变化。
NONIN（3 个）：
➢ rSO2_Ch1（左脑氧）、rSO2_Ch2（右脑氧）、rSO2_Ch3（肌肉氧）。
其中，脑氧饱和度的降低与术后谵妄密切相关

OHMEDA（15 个）：
➢ TVexp（呼气末潮气量，是从呼吸机采集到的实际的患者呼出参数，并非医生所设）；
➢ MVexp（呼气末分钟通气量）；
➢ RRtotal（呼吸频率）；
➢ Circuit_O2（吸氧浓度）；
➢ Ppeak（气道峰压）、Pmean（平均气道压）；
➢ MVexp_spont（自主分钟通气量）；
➢ RR_spont（自主呼吸频率）；
➢ TVexp_spont（自主呼吸呼气末潮气量）；
➢ TVinsp（吸气末潮气量）；
➢ MVinsp（吸气末分钟通气量）；
➢ PEEPe_i（外源性、内源性呼气末正压综合，即实际呼气末正压）；
➢ ambient_pressure（环境压力）；
➢ O2_flow、air_flow（氧气、空气在人体呼吸系统中的流动情况）。
以上参数与呼吸相关，而低血压状态往往与呼吸衰竭有关。

PHILIPS（23 个）：
➢ NOM_ECG_CARD_BEAT_RATE（心率）；
➢ NOM_ECG_AMPL_ST_I / ST_II / ST_III / ST_AVR / ST_AVL / ST_AVF / ST_V / 
ST_MCL（不同电极处的心电振幅情况）；
➢ NOM_PULS_OXIM_SAT_O2（脉搏氧饱和度）、NOM_PLETH_PULS_RATE（通过血
氧容积描记波测得的脉率）；
➢ NOM_PRESS_BLD_ART_ABP_MEAN / SYS / DIA（动脉平均压/收缩压/舒张压-ABP
标识）、NOM_PRESS_BLD_VEN_CENT_MEAN（平均压）；
➢ NOM_PULS_RATE（脉率）、NOM_PULS_OXIM_PERF_REL（关联脉搏、血氧饱和度
与血液灌注）；
➢ NOM_AWAY_CO2_ET（呼气末二氧化碳）、NOM_CONC_AWAY_O2_ET（呼气末氧
浓度）、NOM_CONC_AWAY_O2_INSP（吸入氧浓度）；
➢ NOM_CONC_AWAY_SEVOFL_ET / INSP（呼气末/吸入七氟烷浓度）；
➢ NOM_TEMP_NASOPH（鼻咽温度）；
➢ 以上参数与心电、脉搏、血压有关，可评估患者多种生理功能是否正常。

LIDCO（17 个）：
➢ CO（心排量）、CI（心排指数）、SV（每搏射血量）、SI（每搏指数）、HR（心率）；
➢ SVR（外周血管阻力）、SVRI（外周血管阻力指数）；
➢ MAP（平均动脉压）、SYS（收缩压）、DIA（舒张压）；
➢ SVV（每搏量变异度）、PPV（脉压变异率）、SPV（收缩压变异度，恒为 0 或极小的数）、
HRV（心率变化）、PP（压力变异度）；
➢ DO2（氧供，恒为 0）、DO2I（氧供指数，恒为 0）。
以上参数反映心脏功能及其稳定性。
