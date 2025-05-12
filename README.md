# Dynamic network entropy (DNE) for pinpointing the pre-outbreak stage of infectious disease
 Infectious disease outbreaks have the potential to result in substantial human casualties and financial losses. Issuing timely warnings and taking appropriate measures before infectious disease outbreaks can effectively hinder or even prevent the spread of epidemics. However, the spread of infectious diseases is a complex and dynamic process that involves both biological and social systems. Consequently, accurately predicting the onset of infectious disease outbreaks in real-time poses a significant challenge. In this study, we have developed a computational approach called dynamic network entropy (DNE) by constructing city networks and leveraging extensive hospital visit record data to pinpoint early warning signals for infectious disease outbreaks. Specifically, the proposed method can accurately identify pre-outbreaks of two infectious diseases including influenza and hand, foot, and mouth disease (HFMD). The predicted early warning signals preceded the outbreaks or initial peaks by at least 6 weeks for influenza and 5 weeks for HFMD. Therefore, by harnessing detailed dynamic and high-dimensional information, our DNE method presents an innovative strategy for identifying the critical point or pre-outbreaks stage prior to the catastrophic transition into a pandemic outbreak, which holds significant potential for application in the field of public health surveillance.

## Advantages of the Dynamic Network Entropy (DNE) Method

1. **Early Warning Capability**  
   - Unlike conventional methods that detect only the outbreak phase, DNE can identify the **pre-outbreak stage** of infectious diseases.

2. **Data-Driven and Model-Free**  
   - Does not rely on epidemic transmission models or parameter training.
   - Utilizes only **statistical indicators** such as Pearson correlation coefficient and standard deviation.

3. **Utilizes Readily Available Data**  
   - Requires only the **topological structure of district networks** and **weekly new case data**.
   - Leverages **horizontal high-dimensional data** and **macro-scale city networks**.

4. **Effective Real-Time Surveillance**  
   - Successfully detects pre-outbreak stages for diseases like influenza and HFMD, as demonstrated using historical data from Tokyo (2011–2020 for influenza, 2014–2023 for HFMD).
   - Shows potential for **real-time disease monitoring**.

5. **High Performance Compared to Existing Methods**  
   - Demonstrates **superior performance** in identifying critical warning signals before outbreaks.

## About data
The data files used in the paper are in the folder: ./data

### 📊 Virtual Weekly New Cases Data 
| Week | Chiyoda City | Chuo City | Minato City | Shinjuku City | Shibuya City | Hachioji City | Tachikawa City | Nerima City | Suginami City | Toshima City | Koto City | Edogawa City | Itabashi City | Adachi City | Nakano City | Setagaya City | Shinagawa City | Sumida City | Arakawa City | Bunkyo City | Taito City | Kita City | Nerima City |
|------|--------------|-----------|--------------|----------------|---------------|----------------|-----------------|---------------|----------------|----------------|------------|---------------|------------------|----------------|---------------|------------------|------------------|---------------|----------------|----------------|---------------|---------------|
| 2025.1 | 3 | 5 | 4 | 7 | 6 | 3 | 5 | 8 | 5 | 7 | 3 | 7 | 2 | 4 | 2 | 6 | 4 | 9 | 6 | 6 | 8 | 4 | 5 |
| 2025.2 | 4 | 4 | 5 | 6 | 5 | 2 | 6 | 7 | 3 | 6 | 4 | 6 | 3 | 5 | 3 | 5 | 5 | 7 | 5 | 4 | 7 | 3 | 4 |
| 2025.3 | 2 | 6 | 3 | 8 | 7 | 5 | 4 | 6 | 4 | 8 | 5 | 5 | 4 | 3 | 4 | 4 | 6 | 6 | 3 | 7 | 6 | 5 | 6 |
| 2025.4 | 5 | 3 | 6 | 5 | 4 | 4 | 7 | 3 | 7 | 5 | 6 | 4 | 5 | 6 | 5 | 3 | 7 | 3 | 4 | 4 | 9 | 6 | 3 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

> ✅ Notes:
> - All data is **randomly generated** for demonstration purposes, But all the data in the folder are official data released by the authorities.

**At present, the paper is in the external review state. We have only made public the code of the algorithm part involved in the paper. Once the paper is accepted, the complete drawing code will be updated for replication.** 

