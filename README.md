# Dynamic network entropy (DNE) for pinpointing the pre-outbreak stage of infectious disease
## Algorithm:
- The main algorithms of the DNE method are implemented in `Shang_Tokyo_FLU.py` (Influenza Outbreak Early Warning) and `shang_Tokyo_HFMD.py` (HFMD Outbreak Early Warning).
- The comparative SP-DNM and MST-DNM algorithms are located in `HFMD_SP_DMN.py` and `HFMD_MST_DMN.py` respectively.
- The Time-window Entropy algorithm is stored in the folder `./TWS`.
- The data and algorithms for simulation experiments are stored in the folder `./Simulation_Experiment`.

**At present, the paper is under review. We have only made public the code of the algorithm part involved in the paper. Once the paper is accepted, the complete drawing code will be updated for replication.**


## About data
The data files used in the paper are in the folder: ./data

### 📊 Virtual Weekly New Cases Data 
| Week     | Chiyoda City | Chuo City | Minato City | Shinjuku City | Shibuya City | Hachioji City | Tachikawa City | Nerima City | Suginami City | Toshima City | Koto City | Edogawa City | Itabashi City | Adachi City | Nakano City | Setagaya City | Shinagawa City | Sumida City | Arakawa City | Bunkyo City | Taito City | Kita City |
|----------|--------------|-----------|-------------|----------------|---------------|----------------|-----------------|-------------|----------------|----------------|------------|---------------|------------------|----------------|---------------|------------------|------------------|---------------|----------------|----------------|------------|------------|
| 2025.1   | 3            | 5         | 4           | 7              | 6             | 3              | 5               | 8           | 5              | 7              | 3          | 7             | 2                | 4              | 2             | 6                | 4                | 9             | 6              | 6              | 8             | 4           |
| 2025.2   | 4            | 4         | 5           | 6              | 5             | 2              | 6               | 7           | 3              | 6              | 4          | 6             | 3                | 5              | 3             | 5                | 5                | 7             | 5              | 4              | 7             | 3           |
| 2025.3   | 2            | 6         | 3           | 8              | 7             | 5              | 4               | 6           | 4              | 8              | 5          | 5             | 4                | 3              | 4             | 4                | 6                | 6             | 3              | 7              | 6             | 5           |
| 2025.4   | 5            | 3         | 6           | 5              | 4             | 4              | 7               | 3           | 7              | 5              | 6          | 4             | 5                | 6              | 5             | 3                | 7                | 3             | 4              | 4              | 9             | 6           |
| ...      | ...          | ...       | ...         | ...            | ...           | ...            | ...             | ...         | ...            | ...            | ...        | ...           | ...              | ...            | ...           | ...              | ...              | ...           | ...            | ...            | ...           | ...           |

> ✅ Notes:
> - All data is **randomly generated** for demonstration purposes, But all the data in the folder are official data released by the authorities.
> - We are grateful for the suggestions from all the anonymous reviewers for the improvement of this article.


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