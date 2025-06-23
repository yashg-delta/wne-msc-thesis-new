# Informer In Algorithmic Investment Strategies on High Frequency Bitcoin Data

**Filip Stefaniuk¹ and Robert Ślepaczuk²**

¹University of Warsaw, Faculty of Economic Sciences, Ul. Długa 44/50, 00-241 Warsaw, Poland, ORCID: https://orcid.org/0009-0004-4968-8704, email: filip.stefaniuk@gmail.com
²University of Warsaw, Faculty of Economic Sciences, Department of Quantitative Finance and Machine Learning, Quantitative Finance Research Group, Ul. Długa 44/50, 00-241 Warsaw, Poland, ORCID: https://orcid.org/0000-0001-5527-2014, Corresponding author: rslepaczuk@wne.uw.edu.pl

### Abstract

[cite_start]The article investigates the usage of Informer architecture for building automated trading strategies for high frequency Bitcoin data.  [cite_start]Three strategies using the Informer model with different loss functions: Root Mean Squared Error (RMSE), Generalized Mean Absolute Directional Loss (GMADL) and Quantile loss, are proposed and evaluated against the Buy and Hold benchmark and two benchmark strategies based on technical indicators.  [cite_start]The evaluation is conducted using data of various frequencies: 5 minute, 15 minute, and 30 minute intervals, over the 6 different periods.  [cite_start]The performance of the model using RMSE loss worsens when used with higher frequency data while the model that uses novel GMADL loss function is benefiting from higher frequency data and when trained on 5 minute interval it beat all the other strategies on most of the testing periods.  [cite_start]The primary contribution of this study is the application and assessment of the RMSE, GMADL and Quantile loss functions with the Informer model to forecast future returns, subsequently using these forecasts to develop automated trading strategies.  [cite_start]The research provides evidence that employing an Informer model trained with the GMADL loss function can result in superior trading outcomes compared to the buy-and-hold approach. 

[cite_start]**Keywords**: Machine Learning, Financial Series Forecasting, Automated Trading Strategy, Informer, Transformer, Bitcoin, High-Frequency Trading, Statistics, GMADL 

[cite_start]**JEL Codes**: C4, C14, C45, C53, C58, G13 

### 1 Introduction

[cite_start]The study explores the idea of building an automated trading strategy for Bitcoin.  [cite_start]Five strategies are proposed and evaluated on the historical Bitcoin data of high frequencies: 5 minutes, 15 minutes, and 30 minutes; from a period of 21.08.2019 to 24.07.2024.  [cite_start]The other three employ the Informer (Zhou et al. 2021), a state-of-the-art attention-based neural network model designed to efficiently handle long time series, to predict the returns and subsequently choose positions according to the model's forecasts.  The work aims to answer the following research questions:

[cite_start]Q: Is it possible to create an algorithmic strategy for trading Bitcoin, that is more efficient than the Buy& Hold approach? 
[cite_start]Q: Does signal from Informer model allow to create strategies that are more efficient on trading Bitcoin than strategies based on technical indicators? 
[cite_start]Q: How does selection of the machine learning model loss function influence the strategy performance? 
[cite_start]Q: Does usage of higher frequency data allow to create more efficient strategies? 

[cite_start]To the best of current knowledge, no other research has yet been performed where an Informer model is trained with the Quantile or GMADL (Michańków et al. 2024) loss function, followed by the utilization of its forecasts in buy/sell signals generation to develop automated trading strategies. 

### 3 Data

[cite_start]The data used in the research consider the BTC/USDT cryptocurrency pair from a period of 21.08.2019 to 24.07.2024 (5 years).  [cite_start]The research uses k-line intervals of 5min, 15min and 30min. 

![Figure 1: Price of the BTC/USDT](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_6_image_0.png)
*Note: Price of BTC/USDT cryptocurrency pair in a period from 21.08.2019 to 24.07.2024. The data was obtained from https://www.binance.com/en-NG/landing/data.* 

Then, returns from each interval are computed as:
[cite_start]$$returns = \frac{close\ price - open\ price}{open\ price} \quad (1)$$ 

| Statistic | BTC/USDT 5min |
| :--- | :--- |
| count | 518400 |
| mean | 0.0000060 |
| std | 0.0021843 |
| min | -0.1022537 |
| 25% percentile | -0.0007716 |
| 50% percentile | 0 |
| 75% percentile | 0.0007855 |
| max | 0.1842885 |
| kurtosis | 203.12 |
| skewness | 0.57 |
| KS test stat. | 0.49 |
| KS test p-value | 0.00e+00 |
[cite_start]*Table 1: Descriptive statistics of BTC/USDT* 
*Note: Descriptive statistics of returns with intervals of 5min, 15min and 30min. The statistics are not annualized. Null hypothesis of Kolmogorov-Smirnov (KS) test is that the distribution is normal.* 

[cite_start]The given analysis underscores the significance of investigating and creating trading systems that function in shorter time intervals.  [cite_start]This is because rapid and abrupt price movements, which are critical for effective trading strategies, would go unnoticed by algorithms that operate on longer intervals. 

#### 3.1 Additional Data

[cite_start]The data was enhanced with additional information that attempts to capture this information.  Additional data that was added composes of:

* [cite_start]The Cboe Volatility Index (VIX Index) 
* [cite_start]The Federal Funds effective rates 
* [cite_start]Crypto Fear/Greed index 

#### 3.2 Data Windows

The research follows the approach of Michańków et al. [cite_start]2022, where strategies are independently evaluated on a rolling window that passes through the testing period.  [cite_start]A rolling window consists of an in sample part of the 24 months (2 years) and out of sample part of 6 months.  [cite_start]In total six windows with identical number of data points were created.  [cite_start]The out of sample is used for testing, further referenced as the test part.  [cite_start]The in sample part is split into train and validation parts, with the validation part being 20% of the in sample data. 

![Figure 5: Rolling data windows](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_9_image_0.png)
*Note: The figure presents how the dataset was split into rolling data windows consisting of in sample and out of sample parts.*

| Part | BTC/USDT 5min |
| :--- | :--- |
| train | 165888 |
| validation | 41472 |
| test | 51840 |
[cite_start]*Table 2: Number of data points* 

| Statistic | W1 | W2 | W3 | W4 | W5 | W6 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| count | 51840 | 51840 | 51840 | 51840 | 51840 | 51840 |
| mean | 0.0000010 | -0.0000089 | 0.0000016 | 0.0000050 | 0.0000080 | 0.0000099 |
| std | 0.0023011 | 0.0021663 | 0.0015967 | 0.0013956 | 0.0013791 | 0.0016259 |
| min | -0.0989054 | -0.0438821 | -0.0469833 | -0.0263642 | -0.0866874 | -0.0370693 |
| 25% percentile | -0.0010410 | -0.0009298 | -0.0005471 | -0.0005338 | -0.0004952 | -0.0006930 |
| 50% percentile | -0.0000139 | -0.0000029 | 0.0000070 | 0 | 0 | 0.0000064 |
| 75% percentile | 0.0010127 | 0.0009137 | 0.0005638 | 0.0005350 | 0.0005185 | 0.0007262 |
| max | 0.0426144 | 0.0643117 | 0.0487321 | 0.0200162 | 0.0501479 | 0.0189663 |
| kurtosis | 124.60 | 27.33 | 72.17 | 27.74 | 419.46 | 18.82 |
| skewness | -1.41 | 0.61 | -0.27 | -0.27 | -3.84 | -0.59 |
| KS test stat. | 0.49 | 0.49 | 0.50 | 0.50 | 0.50 | 0.50 |
| KS test p-value | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
[cite_start]*Table 3: Descriptive statistics for 5 min out-of-sample data* 
*Note: Descriptive statistics of returns in data windows of 5min interval data. The statistics are not annualized. [cite_start]Null hypothesis of Kolmogorov-Smirnov (KS) test in that the distribution is normal.* 

![Figure 6: Distributions of returns for out-of-sample data](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_11_image_0.png)
*Note: Visualizations of the returns distributions for the out-of-sample data for different windows from testing period, for 5min. intervals. On the left, heatmaps visualize distances between each pair of distributions according to Wasserstein distance. [cite_start]On the right distributions are visualized as violin plots.* 

### 4 Methodology

#### 4.1 Evaluation Framework

[cite_start]Formally, a trading strategy is a function $s_θ : R^{λ×d} → \{−1, 0, 1\}$ where θ are strategy hyperparameters, λ is the size of the lookback window and d is the dimensionality of the vector $x_t ∈ R^d$ that represents new, known and relevant information at each time step t, which can be used by the strategy.  [cite_start]The codomain values {−1, 0, 1} respectively, represent: short position, no position, and long position. 

[cite_start]The following metrics are computed:

* [cite_start]Annualized Return Compounded (ARC) 
* [cite_start]Annualized Standard Deviation (ASD) 
* [cite_start]Information Ratio (IR*) 
* [cite_start]Maximum Drawdown (MD) 
* [cite_start]Modified Information Ratio (IR**) 
* Number of trades (N) 
* [cite_start]Percentage of Long Position (LONG) / Short Position (SHORT) 

#### 4.2 Strategies

##### 4.2.4 Informer based strategies

[cite_start]The other loss function considered is a Generalized Mean Absolute Directional Loss (GMADL) (Michańków et al. 2024).  [cite_start]It puts more emphasis on the direction of the returns, i.e. whether they were positive or negative rather than precision and rewards the model for correctly predicting larger return values.  The loss function is defined as
[cite_start]$$GMADL = \frac{1}{N} \sum_{i=1}^{N} (-1) \cdot \left(\frac{1}{1+e^{-a \cdot y \cdot \hat{y}}} - \frac{1}{2}\right) \cdot (|y|)^b \quad (35)$$ 
[cite_start]where y is observed value, $\hat{y}$ is model prediction, N is number of observations.  [cite_start]a and b are loss function parameters that control the steepness of the function slope.  [cite_start]In the study they are considered to be equal to a = 100 and b = 2. 

![Figure 7: RMSE vs GMADL loss functions.](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_17_image_0.png)
[cite_start]*Note: Comparison of the two lose functions: Mean Absolute Error (MAE) on the left and Generalized Mean Absolute Directional Loss (GMADL) on the right.* 

[cite_start]**GMADL Informer strategy**: Strategy that uses predictions of the Informer trained with GMADL is defined analogously to the one using RMSE: four threshold values are defined when to enter long, exit long, enter short and exit short and the thresholds are compared directly with the return predicted by the model. 
[cite_start]$$s_{GMADL}(\cdot) = \begin{cases} 1 & \text{if } \hat{y}_t \geq \text{enter long} \\ 0 & \text{if } \hat{y}_t \leq \text{exit long and } p_{t-1}=1 \\ -1 & \text{if } \hat{y}_t \leq \text{enter short} \\ 0 & \text{if } \hat{y}_t \geq \text{exit short and } p_{t-1}=-1 \\ p_{t-1} & \text{else} \end{cases} \quad (38)$$ 
[cite_start]The strategy hyperparameters are $\theta_{GMADL}=(\Theta_{GMADL}$ enter long, exit long, enter short, exit short) and OGMADL are parameters of the Informer model trained with GMADL loss. 

### 5 Experiments

#### 5.1 Training and hyperparameter selection

##### 5.1.3 Informer Strategies training and hyperparameters

[cite_start]**Training and model hyperparameters** A separate instance of Informer was trained for each data window and for each RMSE, Quantile and GMADL loss functions.  [cite_start]The target variable in each case was returns. 

| Parameter | 5min |
| :--- | :--- |
| | **GMADL Informer** |
| past window | 28 |
| batch size | 256 |
| model dimensionality (d) | 256 |
| fully connected layer dim | 256 |
| attention heads h | 2 |
| dropout | 0.01 |
| number of encoder layers | 1 |
| number of decoder layers | 3 |
| learning rate | 0.0001 |
[cite_start]*Table 12: Selected hyperparameters for Informer model.* 
[cite_start]*Note: Table presents hyperparameter values selected for Informer model trained on 5min, 15min and 30min interval data with RMSE loss, Quantile Loss and GMADL loss functions* 

[cite_start]**GMADL Informer strategy hyperparameters** The last strategy was created using the Informer trained with GMADL loss funtion.  [cite_start]The hyperparameters of this strategy are enter long, exit long, enter short and exit short. 

| Table Window | enter long | exit Long | enter Short | exit Short |
| :--- | :--- | :--- | :--- | :--- |
| **W1-5min** | 0.004 | | -0.005 | |
| **W2-5min** | 0.002 | | -0.001 | |
| **W3-5min** | | | -0.006 | 0.003 |
| **W4-5min** | 0.002 | | -0.005 | |
| **W5-5min** | 0.002 | | -0.003 | |
| **W6-5min** | 0.001 | | -0.007 | |
[cite_start]*Table 18: Selected hyperparameter values for GMADL Informer Strategy* 
[cite_start]*Note: Table presents hyperparameter values selected for the GMADL Informer strategy for each window of 5min, 15min and 30min interval data.* 

#### 5.2 Evaluation Results

##### 5.2.3 Evaluation on 5 min data

[cite_start]The evaluation results for the 5 min interval throughout the testing period are presented in Figure 12, a GMADL Informer strategy significantly outperformed the other strategies.  [cite_start]Throughout the testing period, it achieved annualized returns of 115%, while maintaining annualized standard division at a level similar to the buy-and-hold benchmark.  [cite_start]The maximum drawdown of this strategy was only 32.7% which is significantly lower than the 77.3% of Buy and Hold. 

![Figure 12: Evaluation results on 5 min data](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_30_image_0.png)

| Strategy | VAL | ARC | ASD | IR* | MD | IR** | N | LONG | SHORT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Buy and Hold | 1.441 | 13.14% | 57.74% | 0.228 | 77.31% | 0.039 | 2 | 100.00% | 0.00% |
| GMADL Informer | 9.747 | 115.88% | 54.44% | 2.129 | 32.66% | 7.552 | 846 | 44.80% | 41.51% |
*Note: Evaluation results on the whole testing period of 5min interval BTC/USDT data. The presented metrics are portfolio value at the end of the evaluation period (VAL), Annualized Return Compound (ARD), Annualized Standard Deviation (ASD), Information Ratio (IR), Maximum Drawdown (MD), Modified Information Ratio (IR**), number of trades (N) and percent of the long/short positions (LONG/SHORT).* 

Looking at the evaluation results for each of the data windows separately, presented in Figure 13, confirms the superiority of the GMADL informer strategy.  It was the top performing strategy on all the evaluation periods but the last one, on which the best performing strategy was Buy and Hold, suggesting this was the most difficult period to trade. 

![Figure 13: Evaluation results for individual windows on 5 min data](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_31_image_0.png)

##### 5.2.4 Best strategies

The strategies that performed better than the buy-and-hold benchmark are again presented in Figure 14. 

![Figure 14: Best strategies](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_32_image_0.png)

| Strategy | VAL | ARC | ASD | IR* | MD | IR** | N | LONG | SHORT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Buy and Hold | 1.441 | 0.131 | 0.577 | 0.228 | 0.773 | 0.039 | 2 | 100.00% | 0.00% |
| GMADL Informer (5min) | 9.747 | 1.159 | 0.544 | 2.129 | 0.327 | 7.552 | 846 | 44.80% | 41.51% |
*Note: Strategies that achieved better performance than the buy-and-hold benchmark evaluated on testing period with BTC/USDT data. The presented metrics are. portfolio value at the end of the evaluation period (VAL), Annualized Return Compound (ARD), Annualized Standard Deviation (ASD). [cite_start]Information Ratio (IR), Maximum Drawdown (MD), Modified Information Ratio (IR**), number of trades (N) and percent of the long/short positions (LONG/SHORT).* 

GMADL Informer strategy seems to be benefiting from the higher frequency data, with the IR** value monotonically increasing when the data frequency is increased. 

[cite_start]To verify statistical significance of the results, a probabilistic t-test that compares the Information Ratios of the best strategies against the buy-and-hold benchmark was conducted. 

| Strategy | N | σ | t-statistic | p-value |
| :--- | :--- | :--- | :--- | :--- |
| GMADL (5min) | 311040 | 2.820834 | 375.84 | 0.000000*** |
*Table 19: Statistical t-test for comparing the performance of strategies over buy-and-hold.* 
*Note: T-test HO: The information Ratio of the strategy is not greater than the buy-and-hold information ratio. [cite_start]The values marked with *** indicate the p-value lower than the critical value 0.01, rejecting the null hypothesis.* 

### 6 Sensitivity Analysis

[cite_start]This section presents a brief analysis of how factors that impact the selection of strategy hyperparameters influence the strategy performance. 

#### 6.1 Validation part size

[cite_start]The analysis has been carried out only for the best strategies, that is GMADL Informer with 5min data.  [cite_start]The results for evaluating GMADL Informer on 5 minute interval dataset when selecting hyperparameters using various validation part sizes are presented on Figure 15.  [cite_start]For this strategy, selecting a longer or shorter validation period than 6 months resulted in decreased strategy performance.  [cite_start]However, regardless of the length of the validation part, the strategy outperformed the benchmark, and Modified Information Ratio stayed on relatively high level. 

![Figure 15: GMADL Informer strategies with parameters selected using different lengths of validation windows](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_34_image_0.png)

| Strategy | VAL | ARC | ASD | IR* | MD | IR** | N | LONG | SHORT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Buy and Hold | 1.441 | 13.1% | 57.7% | 0.228 | 77.3% | 0.039 | 2 | 100.0% | 0.0% |
| 6 months | 9.747 | 115.9% | 54.4% | 2.129 | 32.7% | 7.552 | 846 | 44.8% | 41.5% |
*Note: Best GMADL Informer strategy with hyperparameters selected on various lengths of validation windows. The presented metrics are portfolio value at the end of the evaluation period (VAL), Annualized Return Compound (ARD), Annualized Standard Deviation (ASD), Information Ratio (IR), Maximum Drawdown (MD), Modified Information Ratio (IR**), number of trades (N) and percent of the long/short positions (LONG/SHORT)* 

#### 6.2 Number of data windows

This section explores how the result of the best strategy - GMADL Informer with 5 min data is affected, if the testing period is split differently: into three or twelve windows. 

The evaluation results are presented in Figure 16.  It can be seen that the performance of the strategy was worse both when the number of windows increased and decreased.  However, this conclusion should not belittle the fact that the GMADL Informer strategy, evaluated on different numbers of windows, still achieves impressive results compared to the benchmark. 

![Figure 16: GMADL Informer strategies evaluated on different number of windows](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_35_image_0.png)

| Strategy | VAL | ARC | ASD | IR* | MD | IR** | N | LONG | SHORT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Buy and Hold | 1.44 | 13.1% | 57.7% | 0.23 | 77.3% | 0.04 | 2 | 100.0% | 0.0% |
| 6 windows | 9.75 | 115.9% | 54.4% | 2.13 | 32.7% | 7.55 | 846 | 44.8% | 41.5% |
*Note: Best GMADL Informer strategy evaluated on different number of windows. [cite_start]The presented metrics are portfolio value at the end of the evaluation period (VAL), Annualized Return Compound (ARD), Annualized Standard Deviation (ASD), Information Ratio (IR), Maximum Drawdown (MD), Modified Information Ratio (IR**), number of trades (N) and percent of the long/short positions (LONG/SHORT).* 

#### 6.3 Top n-th strategy

Figure 17 present the modified information ratio after evaluating strategies with the top 10 hyper-parameter sets from validation windows.  In case of GMADL Strategy with 5 min data, the best overall performing strategy is indeed the one with the first set of hyperparameters, and the consecutive sets gradually decrease the results of the strategy. 

![Figure 17: Strategies with top 10 hyperparameter combinations for 5min data](https://storage.googleapis.com/assistive-research/images/24-06-18/143132-8x9q/2503_18096v1_page_36_image_0.png)
*Note: Modified Information Ratio(IR**) of the strategies evaluated on the whole testing period, with the strategies using the top 10 hyperparameter sets according to IR from the evaluation on the validation set.* 

### 7 Conclusion

[cite_start]The research aimed to explore different methods to create automated investment strategies for trading Bitcoin.  [cite_start]The strategies were evaluated independently over six periods on BTC/USDT data of various frequencies: 5 minutes, 15 minutes, and 30 minutes. 

[cite_start]The best performing strategy was the one based on predictions of the Informer model trained on 5-minute data with the GMADL loss function.  [cite_start]In addition, it significantly outperformed all the other strategies, proving to be best in almost all testing periods. 

[cite_start]Based on the results of the research an attempt to answer the research questions can be made: 

**Q: Is it possible to create an algorithmic strategy for trading Bitcoin, that is more efficient than Buy&Hold approach?**
[cite_start]The research showed that it is possible to create an algorithmic strategy that outperforms the Buy&Hold benchmark on the Bitcoin data. 

**Q: How does selection of the machine learning model loss function influence the strategy performance?**
[cite_start]Informer trained with GMADL loss function benefited from higher frequency.  [cite_start]From the three tested loss function, GMADL loss function seem to be best fitted for creating models that are to be used in automated trading strategies. 

**Q: Does usage of higher frequency data allow to create more efficient strategies?**
[cite_start]The usage of high frequency data improved the performance of the GMADL Informer strategy, while deteriorated the performance of the strategy based on the Informer with RMSE loss function. 

[cite_start]The main contribution of the study was the analysis of the novel application of loss functions: Quantile and GMADL loss, to train Informer model to predict returns of Bitcoin and compare results with the better established approach of training the machine learning model with RMSE loss.  [cite_start]It showed that the GMADL loss function allows to effectively train the machine learning model, which is able to provide meaningful signal for the trading strategy.  [cite_start]Such a strategy can be deployed to provide higher yields than the buy-and-hold approach.
