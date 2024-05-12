import matplotlib.pyplot as plt
import re

log_output = """ 
INFO:root:TRAIN LOG: steps 0, episodes   0, returns 914.99/915.26/908.34/921.95/5 (mean/median/min/max/num), 366.17 steps/s
2024-03-28 17:41:37,111 | TRAIN LOG: steps 0, episodes   0, returns 914.99/915.26/908.34/921.95/5 (mean/median/min/max/num), 366.17 steps/s
INFO:root:TEST LOG: steps 0, episodes   0, returns 912.41/910.82/906.20/922.21/5 (mean/median/min/max/num), 366.17 steps/s
2024-03-28 17:41:39,816 | TEST LOG: steps 0, episodes   0, returns 912.41/910.82/906.20/922.21/5 (mean/median/min/max/num), 366.17 steps/s
INFO:root:Normalized LOG: steps 0, episodes   0, returns 0.29/0.29/0.29/0.30/5 (mean/median/min/max/num), 366.17 steps/s
2024-03-28 17:41:39,816 | Normalized LOG: steps 0, episodes   0, returns 0.29/0.29/0.29/0.30/5 (mean/median/min/max/num), 366.17 steps/s
INFO:root:TRAIN LOG: steps 1000, episodes   1, returns 730.95/909.22/0.00/921.95/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:42:25,013 | TRAIN LOG: steps 1000, episodes   1, returns 730.95/909.22/0.00/921.95/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 1000, episodes   1, returns 3786.70/4931.29/-301.29/5107.82/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:42:27,543 | TEST LOG: steps 1000, episodes   1, returns 3786.70/4931.29/-301.29/5107.82/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 1000, episodes   1, returns 0.98/1.25/0.01/1.29/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:42:27,543 | Normalized LOG: steps 1000, episodes   1, returns 0.98/1.25/0.01/1.29/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 2000, episodes   2, returns 549.11/908.34/0.00/921.95/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 17:43:12,807 | TRAIN LOG: steps 2000, episodes   2, returns 549.11/908.34/0.00/921.95/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TEST LOG: steps 2000, episodes   2, returns 3833.01/4626.69/1566.00/4775.44/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 17:43:15,222 | TEST LOG: steps 2000, episodes   2, returns 3833.01/4626.69/1566.00/4775.44/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:Normalized LOG: steps 2000, episodes   2, returns 0.99/1.18/0.45/1.21/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 17:43:15,222 | Normalized LOG: steps 2000, episodes   2, returns 0.99/1.18/0.45/1.21/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TRAIN LOG: steps 3000, episodes   3, returns 366.06/0.00/0.00/921.95/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 17:44:00,167 | TRAIN LOG: steps 3000, episodes   3, returns 366.06/0.00/0.00/921.95/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TEST LOG: steps 3000, episodes   3, returns 4896.51/4890.08/4710.51/5117.12/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 17:44:02,640 | TEST LOG: steps 3000, episodes   3, returns 4896.51/4890.08/4710.51/5117.12/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:Normalized LOG: steps 3000, episodes   3, returns 1.24/1.24/1.20/1.29/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 17:44:02,640 | Normalized LOG: steps 3000, episodes   3, returns 1.24/1.24/1.20/1.29/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TRAIN LOG: steps 4000, episodes   4, returns 181.67/0.00/0.00/908.34/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 17:44:47,586 | TRAIN LOG: steps 4000, episodes   4, returns 181.67/0.00/0.00/908.34/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TEST LOG: steps 4000, episodes   4, returns 3106.33/4940.73/246.80/5035.07/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 17:44:49,690 | TEST LOG: steps 4000, episodes   4, returns 3106.33/4940.73/246.80/5035.07/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:Normalized LOG: steps 4000, episodes   4, returns 0.82/1.25/0.14/1.27/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 17:44:49,690 | Normalized LOG: steps 4000, episodes   4, returns 0.82/1.25/0.14/1.27/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TRAIN LOG: steps 5000, episodes   5, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:45:34,832 | TRAIN LOG: steps 5000, episodes   5, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 5000, episodes   5, returns 3690.13/4856.75/970.92/5121.48/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:45:36,715 | TEST LOG: steps 5000, episodes   5, returns 3690.13/4856.75/970.92/5121.48/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 5000, episodes   5, returns 0.95/1.23/0.31/1.30/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:45:36,715 | Normalized LOG: steps 5000, episodes   5, returns 0.95/1.23/0.31/1.30/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 6000, episodes   6, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:46:21,699 | TRAIN LOG: steps 6000, episodes   6, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TEST LOG: steps 6000, episodes   6, returns 4343.59/5133.65/1434.46/5178.93/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:46:24,222 | TEST LOG: steps 6000, episodes   6, returns 4343.59/5133.65/1434.46/5178.93/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:Normalized LOG: steps 6000, episodes   6, returns 1.11/1.30/0.42/1.31/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:46:24,222 | Normalized LOG: steps 6000, episodes   6, returns 1.11/1.30/0.42/1.31/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TRAIN LOG: steps 7000, episodes   7, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:47:09,369 | TRAIN LOG: steps 7000, episodes   7, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 7000, episodes   7, returns 3649.76/5191.79/398.45/5413.37/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:47:11,438 | TEST LOG: steps 7000, episodes   7, returns 3649.76/5191.79/398.45/5413.37/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 7000, episodes   7, returns 0.95/1.31/0.17/1.36/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:47:11,438 | Normalized LOG: steps 7000, episodes   7, returns 0.95/1.31/0.17/1.36/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 8000, episodes   8, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:47:56,595 | TRAIN LOG: steps 8000, episodes   8, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 8000, episodes   8, returns 3085.68/3085.99/-686.86/5325.68/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:47:58,971 | TEST LOG: steps 8000, episodes   8, returns 3085.68/3085.99/-686.86/5325.68/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 8000, episodes   8, returns 0.81/0.81/-0.09/1.34/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:47:58,972 | Normalized LOG: steps 8000, episodes   8, returns 0.81/0.81/-0.09/1.34/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 9000, episodes   9, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.31 steps/s
2024-03-28 17:48:43,789 | TRAIN LOG: steps 9000, episodes   9, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.31 steps/s
INFO:root:TEST LOG: steps 9000, episodes   9, returns 3418.89/5073.30/-875.30/5362.61/5 (mean/median/min/max/num), 22.31 steps/s
2024-03-28 17:48:46,031 | TEST LOG: steps 9000, episodes   9, returns 3418.89/5073.30/-875.30/5362.61/5 (mean/median/min/max/num), 22.31 steps/s
INFO:root:Normalized LOG: steps 9000, episodes   9, returns 0.89/1.28/-0.13/1.35/5 (mean/median/min/max/num), 22.31 steps/s
2024-03-28 17:48:46,031 | Normalized LOG: steps 9000, episodes   9, returns 0.89/1.28/-0.13/1.35/5 (mean/median/min/max/num), 22.31 steps/s
INFO:root:TRAIN LOG: steps 10000, episodes  10, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:49:31,207 | TRAIN LOG: steps 10000, episodes  10, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TEST LOG: steps 10000, episodes  10, returns 4467.37/4897.86/2111.50/5283.27/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:49:33,722 | TEST LOG: steps 10000, episodes  10, returns 4467.37/4897.86/2111.50/5283.27/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:Normalized LOG: steps 10000, episodes  10, returns 1.14/1.24/0.58/1.33/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:49:33,722 | Normalized LOG: steps 10000, episodes  10, returns 1.14/1.24/0.58/1.33/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TRAIN LOG: steps 11000, episodes  11, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.22 steps/s
2024-03-28 17:50:18,731 | TRAIN LOG: steps 11000, episodes  11, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.22 steps/s
INFO:root:TEST LOG: steps 11000, episodes  11, returns 3691.16/4301.98/458.62/5165.97/5 (mean/median/min/max/num), 22.22 steps/s
2024-03-28 17:50:20,717 | TEST LOG: steps 11000, episodes  11, returns 3691.16/4301.98/458.62/5165.97/5 (mean/median/min/max/num), 22.22 steps/s
INFO:root:Normalized LOG: steps 11000, episodes  11, returns 0.96/1.10/0.19/1.31/5 (mean/median/min/max/num), 22.22 steps/s
2024-03-28 17:50:20,717 | Normalized LOG: steps 11000, episodes  11, returns 0.96/1.10/0.19/1.31/5 (mean/median/min/max/num), 22.22 steps/s
INFO:root:TRAIN LOG: steps 12000, episodes  12, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:51:05,895 | TRAIN LOG: steps 12000, episodes  12, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 12000, episodes  12, returns 2916.77/3663.62/219.71/5311.16/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:51:08,325 | TEST LOG: steps 12000, episodes  12, returns 2916.77/3663.62/219.71/5311.16/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 12000, episodes  12, returns 0.77/0.95/0.13/1.34/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:51:08,325 | Normalized LOG: steps 12000, episodes  12, returns 0.77/0.95/0.13/1.34/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 13000, episodes  13, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 17:51:53,460 | TRAIN LOG: steps 13000, episodes  13, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TEST LOG: steps 13000, episodes  13, returns 4814.88/5122.33/3387.63/5295.51/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 17:51:55,818 | TEST LOG: steps 13000, episodes  13, returns 4814.88/5122.33/3387.63/5295.51/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:Normalized LOG: steps 13000, episodes  13, returns 1.22/1.30/0.88/1.34/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 17:51:55,818 | Normalized LOG: steps 13000, episodes  13, returns 1.22/1.30/0.88/1.34/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TRAIN LOG: steps 14000, episodes  14, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:52:40,986 | TRAIN LOG: steps 14000, episodes  14, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TEST LOG: steps 14000, episodes  14, returns 4863.99/5140.33/3594.14/5337.58/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:52:43,493 | TEST LOG: steps 14000, episodes  14, returns 4863.99/5140.33/3594.14/5337.58/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:Normalized LOG: steps 14000, episodes  14, returns 1.23/1.30/0.93/1.35/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:52:43,493 | Normalized LOG: steps 14000, episodes  14, returns 1.23/1.30/0.93/1.35/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TRAIN LOG: steps 15000, episodes  15, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:53:28,660 | TRAIN LOG: steps 15000, episodes  15, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TEST LOG: steps 15000, episodes  15, returns 4439.89/5089.13/3076.02/5288.52/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:53:30,807 | TEST LOG: steps 15000, episodes  15, returns 4439.89/5089.13/3076.02/5288.52/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:Normalized LOG: steps 15000, episodes  15, returns 1.13/1.29/0.81/1.34/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:53:30,807 | Normalized LOG: steps 15000, episodes  15, returns 1.13/1.29/0.81/1.34/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TRAIN LOG: steps 16000, episodes  16, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:54:15,968 | TRAIN LOG: steps 16000, episodes  16, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TEST LOG: steps 16000, episodes  16, returns 3529.21/3779.33/1517.37/5204.55/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:54:17,921 | TEST LOG: steps 16000, episodes  16, returns 3529.21/3779.33/1517.37/5204.55/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:Normalized LOG: steps 16000, episodes  16, returns 0.92/0.98/0.44/1.32/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 17:54:17,921 | Normalized LOG: steps 16000, episodes  16, returns 0.92/0.98/0.44/1.32/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TRAIN LOG: steps 17000, episodes  17, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.18 steps/s
2024-03-28 17:55:03,017 | TRAIN LOG: steps 17000, episodes  17, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.18 steps/s
INFO:root:TEST LOG: steps 17000, episodes  17, returns 2154.74/940.63/206.00/5117.54/5 (mean/median/min/max/num), 22.18 steps/s
2024-03-28 17:55:04,609 | TEST LOG: steps 17000, episodes  17, returns 2154.74/940.63/206.00/5117.54/5 (mean/median/min/max/num), 22.18 steps/s
INFO:root:Normalized LOG: steps 17000, episodes  17, returns 0.59/0.30/0.13/1.29/5 (mean/median/min/max/num), 22.18 steps/s
2024-03-28 17:55:04,609 | Normalized LOG: steps 17000, episodes  17, returns 0.59/0.30/0.13/1.29/5 (mean/median/min/max/num), 22.18 steps/s
INFO:root:TRAIN LOG: steps 18000, episodes  18, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 17:55:49,573 | TRAIN LOG: steps 18000, episodes  18, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:TEST LOG: steps 18000, episodes  18, returns 4875.32/5100.52/3896.82/5313.52/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 17:55:52,072 | TEST LOG: steps 18000, episodes  18, returns 4875.32/5100.52/3896.82/5313.52/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:Normalized LOG: steps 18000, episodes  18, returns 1.24/1.29/1.00/1.34/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 17:55:52,072 | Normalized LOG: steps 18000, episodes  18, returns 1.24/1.29/1.00/1.34/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:TRAIN LOG: steps 19000, episodes  19, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:56:37,064 | TRAIN LOG: steps 19000, episodes  19, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TEST LOG: steps 19000, episodes  19, returns 3897.91/4933.61/39.89/5374.06/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:56:38,961 | TEST LOG: steps 19000, episodes  19, returns 3897.91/4933.61/39.89/5374.06/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:Normalized LOG: steps 19000, episodes  19, returns 1.00/1.25/0.09/1.36/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:56:38,961 | Normalized LOG: steps 19000, episodes  19, returns 1.00/1.25/0.09/1.36/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TRAIN LOG: steps 20000, episodes  20, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:57:23,944 | TRAIN LOG: steps 20000, episodes  20, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TEST LOG: steps 20000, episodes  20, returns 583.83/445.61/-980.19/1587.18/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:57:24,963 | TEST LOG: steps 20000, episodes  20, returns 583.83/445.61/-980.19/1587.18/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:Normalized LOG: steps 20000, episodes  20, returns 0.22/0.18/-0.16/0.45/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 17:57:24,964 | Normalized LOG: steps 20000, episodes  20, returns 0.22/0.18/-0.16/0.45/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TRAIN LOG: steps 21000, episodes  21, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:58:10,155 | TRAIN LOG: steps 21000, episodes  21, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 21000, episodes  21, returns 3752.85/4572.40/983.40/5383.18/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:58:11,938 | TEST LOG: steps 21000, episodes  21, returns 3752.85/4572.40/983.40/5383.18/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 21000, episodes  21, returns 0.97/1.16/0.31/1.36/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 17:58:11,939 | Normalized LOG: steps 21000, episodes  21, returns 0.97/1.16/0.31/1.36/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 22000, episodes  22, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:58:57,093 | TRAIN LOG: steps 22000, episodes  22, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 22000, episodes  22, returns 1202.44/1015.85/-28.72/3027.71/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:58:58,482 | TEST LOG: steps 22000, episodes  22, returns 1202.44/1015.85/-28.72/3027.71/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 22000, episodes  22, returns 0.36/0.32/0.07/0.80/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 17:58:58,482 | Normalized LOG: steps 22000, episodes  22, returns 0.36/0.32/0.07/0.80/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 23000, episodes  23, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 17:59:43,607 | TRAIN LOG: steps 23000, episodes  23, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TEST LOG: steps 23000, episodes  23, returns 4846.14/5172.51/3301.45/5364.97/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 17:59:46,112 | TEST LOG: steps 23000, episodes  23, returns 4846.14/5172.51/3301.45/5364.97/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:Normalized LOG: steps 23000, episodes  23, returns 1.23/1.31/0.86/1.35/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 17:59:46,112 | Normalized LOG: steps 23000, episodes  23, returns 1.23/1.31/0.86/1.35/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TRAIN LOG: steps 24000, episodes  24, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 18:00:31,094 | TRAIN LOG: steps 24000, episodes  24, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TEST LOG: steps 24000, episodes  24, returns 5012.56/5073.45/4864.31/5150.74/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 18:00:33,587 | TEST LOG: steps 24000, episodes  24, returns 5012.56/5073.45/4864.31/5150.74/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:Normalized LOG: steps 24000, episodes  24, returns 1.27/1.28/1.23/1.30/5 (mean/median/min/max/num), 22.23 steps/s
2024-03-28 18:00:33,587 | Normalized LOG: steps 24000, episodes  24, returns 1.27/1.28/1.23/1.30/5 (mean/median/min/max/num), 22.23 steps/s
INFO:root:TRAIN LOG: steps 25000, episodes  25, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:01:18,518 | TRAIN LOG: steps 25000, episodes  25, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:TEST LOG: steps 25000, episodes  25, returns 3604.20/3304.55/2433.97/4963.19/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:01:20,292 | TEST LOG: steps 25000, episodes  25, returns 3604.20/3304.55/2433.97/4963.19/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:Normalized LOG: steps 25000, episodes  25, returns 0.93/0.86/0.66/1.26/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:01:20,292 | Normalized LOG: steps 25000, episodes  25, returns 0.93/0.86/0.66/1.26/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:TRAIN LOG: steps 26000, episodes  26, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:02:05,213 | TRAIN LOG: steps 26000, episodes  26, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:TEST LOG: steps 26000, episodes  26, returns 3841.08/5010.80/926.10/5340.93/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:02:07,074 | TEST LOG: steps 26000, episodes  26, returns 3841.08/5010.80/926.10/5340.93/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:Normalized LOG: steps 26000, episodes  26, returns 0.99/1.27/0.30/1.35/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:02:07,074 | Normalized LOG: steps 26000, episodes  26, returns 0.99/1.27/0.30/1.35/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:TRAIN LOG: steps 27000, episodes  27, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:02:51,985 | TRAIN LOG: steps 27000, episodes  27, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:TEST LOG: steps 27000, episodes  27, returns 3924.83/5045.53/-652.72/5304.77/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:02:54,460 | TEST LOG: steps 27000, episodes  27, returns 3924.83/5045.53/-652.72/5304.77/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:Normalized LOG: steps 27000, episodes  27, returns 1.01/1.28/-0.08/1.34/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:02:54,460 | Normalized LOG: steps 27000, episodes  27, returns 1.01/1.28/-0.08/1.34/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:TRAIN LOG: steps 28000, episodes  28, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.12 steps/s
2024-03-28 18:03:39,661 | TRAIN LOG: steps 28000, episodes  28, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.12 steps/s
INFO:root:TEST LOG: steps 28000, episodes  28, returns 1449.12/288.78/141.43/4893.83/5 (mean/median/min/max/num), 22.12 steps/s
2024-03-28 18:03:40,930 | TEST LOG: steps 28000, episodes  28, returns 1449.12/288.78/141.43/4893.83/5 (mean/median/min/max/num), 22.12 steps/s
INFO:root:Normalized LOG: steps 28000, episodes  28, returns 0.42/0.15/0.11/1.24/5 (mean/median/min/max/num), 22.12 steps/s
2024-03-28 18:03:40,930 | Normalized LOG: steps 28000, episodes  28, returns 0.42/0.15/0.11/1.24/5 (mean/median/min/max/num), 22.12 steps/s
INFO:root:TRAIN LOG: steps 29000, episodes  29, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:04:26,076 | TRAIN LOG: steps 29000, episodes  29, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 29000, episodes  29, returns 1413.68/633.25/35.00/4248.77/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:04:26,889 | TEST LOG: steps 29000, episodes  29, returns 1413.68/633.25/35.00/4248.77/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 29000, episodes  29, returns 0.41/0.23/0.09/1.09/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:04:26,889 | Normalized LOG: steps 29000, episodes  29, returns 0.41/0.23/0.09/1.09/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 30000, episodes  30, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 18:05:11,931 | TRAIN LOG: steps 30000, episodes  30, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TEST LOG: steps 30000, episodes  30, returns 1641.28/1367.91/193.84/3240.24/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 18:05:12,812 | TEST LOG: steps 30000, episodes  30, returns 1641.28/1367.91/193.84/3240.24/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:Normalized LOG: steps 30000, episodes  30, returns 0.47/0.40/0.12/0.85/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 18:05:12,812 | Normalized LOG: steps 30000, episodes  30, returns 0.47/0.40/0.12/0.85/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TRAIN LOG: steps 31000, episodes  31, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:05:57,952 | TRAIN LOG: steps 31000, episodes  31, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 31000, episodes  31, returns 1545.30/1057.42/39.37/4631.18/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:06:00,174 | TEST LOG: steps 31000, episodes  31, returns 1545.30/1057.42/39.37/4631.18/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 31000, episodes  31, returns 0.44/0.33/0.09/1.18/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:06:00,174 | Normalized LOG: steps 31000, episodes  31, returns 0.44/0.33/0.09/1.18/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 32000, episodes  32, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:06:45,102 | TRAIN LOG: steps 32000, episodes  32, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:TEST LOG: steps 32000, episodes  32, returns 2789.98/2622.70/698.12/5417.52/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:06:46,497 | TEST LOG: steps 32000, episodes  32, returns 2789.98/2622.70/698.12/5417.52/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:Normalized LOG: steps 32000, episodes  32, returns 0.74/0.70/0.24/1.37/5 (mean/median/min/max/num), 22.26 steps/s
2024-03-28 18:06:46,497 | Normalized LOG: steps 32000, episodes  32, returns 0.74/0.70/0.24/1.37/5 (mean/median/min/max/num), 22.26 steps/s
INFO:root:TRAIN LOG: steps 33000, episodes  33, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.28 steps/s
2024-03-28 18:07:31,385 | TRAIN LOG: steps 33000, episodes  33, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.28 steps/s
INFO:root:TEST LOG: steps 33000, episodes  33, returns 3191.29/4626.37/12.81/5017.52/5 (mean/median/min/max/num), 22.28 steps/s
2024-03-28 18:07:33,032 | TEST LOG: steps 33000, episodes  33, returns 3191.29/4626.37/12.81/5017.52/5 (mean/median/min/max/num), 22.28 steps/s
INFO:root:Normalized LOG: steps 33000, episodes  33, returns 0.84/1.18/0.08/1.27/5 (mean/median/min/max/num), 22.28 steps/s
2024-03-28 18:07:33,032 | Normalized LOG: steps 33000, episodes  33, returns 0.84/1.18/0.08/1.27/5 (mean/median/min/max/num), 22.28 steps/s
INFO:root:TRAIN LOG: steps 34000, episodes  34, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 18:08:18,194 | TRAIN LOG: steps 34000, episodes  34, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TEST LOG: steps 34000, episodes  34, returns 3549.73/5179.52/963.33/5375.61/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 18:08:19,903 | TEST LOG: steps 34000, episodes  34, returns 3549.73/5179.52/963.33/5375.61/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:Normalized LOG: steps 34000, episodes  34, returns 0.92/1.31/0.31/1.36/5 (mean/median/min/max/num), 22.14 steps/s
2024-03-28 18:08:19,903 | Normalized LOG: steps 34000, episodes  34, returns 0.92/1.31/0.31/1.36/5 (mean/median/min/max/num), 22.14 steps/s
INFO:root:TRAIN LOG: steps 35000, episodes  35, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 18:09:05,014 | TRAIN LOG: steps 35000, episodes  35, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:TEST LOG: steps 35000, episodes  35, returns 3102.65/3551.54/369.95/5421.57/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 18:09:06,969 | TEST LOG: steps 35000, episodes  35, returns 3102.65/3551.54/369.95/5421.57/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:Normalized LOG: steps 35000, episodes  35, returns 0.82/0.92/0.17/1.37/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 18:09:06,969 | Normalized LOG: steps 35000, episodes  35, returns 0.82/0.92/0.17/1.37/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:TRAIN LOG: steps 36000, episodes  36, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:09:52,109 | TRAIN LOG: steps 36000, episodes  36, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 36000, episodes  36, returns 4090.46/5053.09/640.23/5524.42/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:09:54,647 | TEST LOG: steps 36000, episodes  36, returns 4090.46/5053.09/640.23/5524.42/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 36000, episodes  36, returns 1.05/1.28/0.23/1.39/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:09:54,647 | Normalized LOG: steps 36000, episodes  36, returns 1.05/1.28/0.23/1.39/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 37000, episodes  37, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 18:10:39,593 | TRAIN LOG: steps 37000, episodes  37, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TEST LOG: steps 37000, episodes  37, returns 1925.75/2072.00/670.82/3220.07/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 18:10:40,557 | TEST LOG: steps 37000, episodes  37, returns 1925.75/2072.00/670.82/3220.07/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:Normalized LOG: steps 37000, episodes  37, returns 0.54/0.57/0.24/0.84/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 18:10:40,557 | Normalized LOG: steps 37000, episodes  37, returns 0.54/0.57/0.24/0.84/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TRAIN LOG: steps 38000, episodes  38, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 18:11:25,687 | TRAIN LOG: steps 38000, episodes  38, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TEST LOG: steps 38000, episodes  38, returns 3817.56/4833.83/1575.65/5311.56/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 18:11:28,224 | TEST LOG: steps 38000, episodes  38, returns 3817.56/4833.83/1575.65/5311.56/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:Normalized LOG: steps 38000, episodes  38, returns 0.99/1.23/0.45/1.34/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 18:11:28,225 | Normalized LOG: steps 38000, episodes  38, returns 0.99/1.23/0.45/1.34/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TRAIN LOG: steps 39000, episodes  39, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:12:13,379 | TRAIN LOG: steps 39000, episodes  39, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 39000, episodes  39, returns 5141.05/5183.02/4562.69/5555.16/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:12:15,823 | TEST LOG: steps 39000, episodes  39, returns 5141.05/5183.02/4562.69/5555.16/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 39000, episodes  39, returns 1.30/1.31/1.16/1.40/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:12:15,823 | Normalized LOG: steps 39000, episodes  39, returns 1.30/1.31/1.16/1.40/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 40000, episodes  40, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:13:01,016 | TRAIN LOG: steps 40000, episodes  40, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 40000, episodes  40, returns 758.92/530.13/-1912.69/3635.55/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:13:02,693 | TEST LOG: steps 40000, episodes  40, returns 758.92/530.13/-1912.69/3635.55/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 40000, episodes  40, returns 0.26/0.20/-0.38/0.94/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:13:02,693 | Normalized LOG: steps 40000, episodes  40, returns 0.26/0.20/-0.38/0.94/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 41000, episodes  41, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:13:47,600 | TRAIN LOG: steps 41000, episodes  41, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:TEST LOG: steps 41000, episodes  41, returns 4771.87/4908.04/3812.92/5302.39/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:13:50,101 | TEST LOG: steps 41000, episodes  41, returns 4771.87/4908.04/3812.92/5302.39/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:Normalized LOG: steps 41000, episodes  41, returns 1.21/1.24/0.98/1.34/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:13:50,101 | Normalized LOG: steps 41000, episodes  41, returns 1.21/1.24/0.98/1.34/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:TRAIN LOG: steps 42000, episodes  42, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:14:35,292 | TRAIN LOG: steps 42000, episodes  42, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 42000, episodes  42, returns 3415.84/4531.27/222.32/5456.42/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:14:36,996 | TEST LOG: steps 42000, episodes  42, returns 3415.84/4531.27/222.32/5456.42/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 42000, episodes  42, returns 0.89/1.15/0.13/1.37/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:14:36,997 | Normalized LOG: steps 42000, episodes  42, returns 0.89/1.15/0.13/1.37/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 43000, episodes  43, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:15:22,179 | TRAIN LOG: steps 43000, episodes  43, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 43000, episodes  43, returns 3516.32/5087.47/351.82/5160.30/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:15:23,938 | TEST LOG: steps 43000, episodes  43, returns 3516.32/5087.47/351.82/5160.30/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 43000, episodes  43, returns 0.91/1.29/0.16/1.30/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 18:15:23,938 | Normalized LOG: steps 43000, episodes  43, returns 0.91/1.29/0.16/1.30/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 44000, episodes  44, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.12 steps/s
2024-03-28 18:16:09,156 | TRAIN LOG: steps 44000, episodes  44, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.12 steps/s
INFO:root:TEST LOG: steps 44000, episodes  44, returns 2189.79/1431.22/135.36/5263.03/5 (mean/median/min/max/num), 22.12 steps/s
2024-03-28 18:16:10,243 | TEST LOG: steps 44000, episodes  44, returns 2189.79/1431.22/135.36/5263.03/5 (mean/median/min/max/num), 22.12 steps/s
INFO:root:Normalized LOG: steps 44000, episodes  44, returns 0.60/0.42/0.11/1.33/5 (mean/median/min/max/num), 22.12 steps/s
2024-03-28 18:16:10,243 | Normalized LOG: steps 44000, episodes  44, returns 0.60/0.42/0.11/1.33/5 (mean/median/min/max/num), 22.12 steps/s
INFO:root:TRAIN LOG: steps 45000, episodes  45, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 18:16:55,349 | TRAIN LOG: steps 45000, episodes  45, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:TEST LOG: steps 45000, episodes  45, returns 4229.04/5243.55/276.21/5422.66/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 18:16:57,371 | TEST LOG: steps 45000, episodes  45, returns 4229.04/5243.55/276.21/5422.66/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:Normalized LOG: steps 45000, episodes  45, returns 1.08/1.32/0.14/1.37/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 18:16:57,371 | Normalized LOG: steps 45000, episodes  45, returns 1.08/1.32/0.14/1.37/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:TRAIN LOG: steps 46000, episodes  46, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.19 steps/s
2024-03-28 18:17:42,444 | TRAIN LOG: steps 46000, episodes  46, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.19 steps/s
INFO:root:TEST LOG: steps 46000, episodes  46, returns 3988.26/4701.51/1786.08/4949.07/5 (mean/median/min/max/num), 22.19 steps/s
2024-03-28 18:17:44,472 | TEST LOG: steps 46000, episodes  46, returns 3988.26/4701.51/1786.08/4949.07/5 (mean/median/min/max/num), 22.19 steps/s
INFO:root:Normalized LOG: steps 46000, episodes  46, returns 1.03/1.20/0.50/1.25/5 (mean/median/min/max/num), 22.19 steps/s
2024-03-28 18:17:44,472 | Normalized LOG: steps 46000, episodes  46, returns 1.03/1.20/0.50/1.25/5 (mean/median/min/max/num), 22.19 steps/s
INFO:root:TRAIN LOG: steps 47000, episodes  47, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:18:29,385 | TRAIN LOG: steps 47000, episodes  47, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:TEST LOG: steps 47000, episodes  47, returns 2550.53/2259.49/1058.04/4903.86/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:18:31,056 | TEST LOG: steps 47000, episodes  47, returns 2550.53/2259.49/1058.04/4903.86/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:Normalized LOG: steps 47000, episodes  47, returns 0.68/0.61/0.33/1.24/5 (mean/median/min/max/num), 22.27 steps/s
2024-03-28 18:18:31,057 | Normalized LOG: steps 47000, episodes  47, returns 0.68/0.61/0.33/1.24/5 (mean/median/min/max/num), 22.27 steps/s
INFO:root:TRAIN LOG: steps 48000, episodes  48, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 18:19:15,991 | TRAIN LOG: steps 48000, episodes  48, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TEST LOG: steps 48000, episodes  48, returns 3879.45/4897.43/10.66/5406.58/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 18:19:17,998 | TEST LOG: steps 48000, episodes  48, returns 3879.45/4897.43/10.66/5406.58/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:Normalized LOG: steps 48000, episodes  48, returns 1.00/1.24/0.08/1.36/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 18:19:17,998 | Normalized LOG: steps 48000, episodes  48, returns 1.00/1.24/0.08/1.36/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TRAIN LOG: steps 49000, episodes  49, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 18:20:03,221 | TRAIN LOG: steps 49000, episodes  49, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TEST LOG: steps 49000, episodes  49, returns 3102.53/2875.34/1275.06/4717.99/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 18:20:04,917 | TEST LOG: steps 49000, episodes  49, returns 3102.53/2875.34/1275.06/4717.99/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:Normalized LOG: steps 49000, episodes  49, returns 0.82/0.76/0.38/1.20/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 18:20:04,917 | Normalized LOG: steps 49000, episodes  49, returns 0.82/0.76/0.38/1.20/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TRAIN LOG: steps 50000, episodes  50, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 18:20:50,137 | TRAIN LOG: steps 50000, episodes  50, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TEST LOG: steps 50000, episodes  50, returns 2658.06/4493.04/-764.10/5161.58/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 18:20:52,741 | TEST LOG: steps 50000, episodes  50, returns 2658.06/4493.04/-764.10/5161.58/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:Normalized LOG: steps 50000, episodes  50, returns 0.71/1.15/-0.10/1.30/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 18:20:52,741 | Normalized LOG: steps 50000, episodes  50, returns 0.71/1.15/-0.10/1.30/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TRAIN LOG: steps 51000, episodes  51, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.31 steps/s
2024-03-28 18:21:37,572 | TRAIN LOG: steps 51000, episodes  51, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.31 steps/s
INFO:root:TEST LOG: steps 51000, episodes  51, returns 1385.47/1576.78/9.55/3083.53/5 (mean/median/min/max/num), 22.31 steps/s
2024-03-28 18:21:38,760 | TEST LOG: steps 51000, episodes  51, returns 1385.47/1576.78/9.55/3083.53/5 (mean/median/min/max/num), 22.31 steps/s
INFO:root:Normalized LOG: steps 51000, episodes  51, returns 0.41/0.45/0.08/0.81/5 (mean/median/min/max/num), 22.31 steps/s
2024-03-28 18:21:38,760 | Normalized LOG: steps 51000, episodes  51, returns 0.41/0.45/0.08/0.81/5 (mean/median/min/max/num), 22.31 steps/s
INFO:root:TRAIN LOG: steps 52000, episodes  52, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.21 steps/s
2024-03-28 18:22:23,779 | TRAIN LOG: steps 52000, episodes  52, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.21 steps/s
INFO:root:TEST LOG: steps 52000, episodes  52, returns 4108.48/5253.37/1244.47/5475.72/5 (mean/median/min/max/num), 22.21 steps/s
2024-03-28 18:22:25,719 | TEST LOG: steps 52000, episodes  52, returns 4108.48/5253.37/1244.47/5475.72/5 (mean/median/min/max/num), 22.21 steps/s
INFO:root:Normalized LOG: steps 52000, episodes  52, returns 1.05/1.33/0.37/1.38/5 (mean/median/min/max/num), 22.21 steps/s
2024-03-28 18:22:25,719 | Normalized LOG: steps 52000, episodes  52, returns 1.05/1.33/0.37/1.38/5 (mean/median/min/max/num), 22.21 steps/s
INFO:root:TRAIN LOG: steps 53000, episodes  53, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 18:23:10,968 | TRAIN LOG: steps 53000, episodes  53, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:TEST LOG: steps 53000, episodes  53, returns 3641.59/4089.80/492.54/5055.14/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 18:23:12,914 | TEST LOG: steps 53000, episodes  53, returns 3641.59/4089.80/492.54/5055.14/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:Normalized LOG: steps 53000, episodes  53, returns 0.94/1.05/0.19/1.28/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 18:23:12,914 | Normalized LOG: steps 53000, episodes  53, returns 0.94/1.05/0.19/1.28/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:TRAIN LOG: steps 54000, episodes  54, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:23:58,210 | TRAIN LOG: steps 54000, episodes  54, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TEST LOG: steps 54000, episodes  54, returns 3668.09/4439.03/1501.78/5147.89/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:24:00,080 | TEST LOG: steps 54000, episodes  54, returns 3668.09/4439.03/1501.78/5147.89/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:Normalized LOG: steps 54000, episodes  54, returns 0.95/1.13/0.43/1.30/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:24:00,081 | Normalized LOG: steps 54000, episodes  54, returns 0.95/1.13/0.43/1.30/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TRAIN LOG: steps 55000, episodes  55, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.05 steps/s
2024-03-28 18:24:45,441 | TRAIN LOG: steps 55000, episodes  55, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.05 steps/s
INFO:root:TEST LOG: steps 55000, episodes  55, returns 5099.47/5179.68/4579.32/5360.20/5 (mean/median/min/max/num), 22.05 steps/s
2024-03-28 18:24:47,937 | TEST LOG: steps 55000, episodes  55, returns 5099.47/5179.68/4579.32/5360.20/5 (mean/median/min/max/num), 22.05 steps/s
INFO:root:Normalized LOG: steps 55000, episodes  55, returns 1.29/1.31/1.17/1.35/5 (mean/median/min/max/num), 22.05 steps/s
2024-03-28 18:24:47,937 | Normalized LOG: steps 55000, episodes  55, returns 1.29/1.31/1.17/1.35/5 (mean/median/min/max/num), 22.05 steps/s
INFO:root:TRAIN LOG: steps 56000, episodes  56, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:25:33,221 | TRAIN LOG: steps 56000, episodes  56, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TEST LOG: steps 56000, episodes  56, returns 1048.65/1114.44/-231.31/2656.73/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:25:34,376 | TEST LOG: steps 56000, episodes  56, returns 1048.65/1114.44/-231.31/2656.73/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:Normalized LOG: steps 56000, episodes  56, returns 0.33/0.34/0.02/0.71/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:25:34,376 | Normalized LOG: steps 56000, episodes  56, returns 0.33/0.34/0.02/0.71/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TRAIN LOG: steps 57000, episodes  57, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 18:26:19,634 | TRAIN LOG: steps 57000, episodes  57, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:TEST LOG: steps 57000, episodes  57, returns 4217.94/5193.74/2477.49/5382.44/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 18:26:21,653 | TEST LOG: steps 57000, episodes  57, returns 4217.94/5193.74/2477.49/5382.44/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:Normalized LOG: steps 57000, episodes  57, returns 1.08/1.31/0.67/1.36/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 18:26:21,653 | Normalized LOG: steps 57000, episodes  57, returns 1.08/1.31/0.67/1.36/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:TRAIN LOG: steps 58000, episodes  58, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:27:06,808 | TRAIN LOG: steps 58000, episodes  58, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 58000, episodes  58, returns 2098.26/1228.88/99.33/5227.44/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:27:08,273 | TEST LOG: steps 58000, episodes  58, returns 2098.26/1228.88/99.33/5227.44/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 58000, episodes  58, returns 0.58/0.37/0.10/1.32/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:27:08,273 | Normalized LOG: steps 58000, episodes  58, returns 0.58/0.37/0.10/1.32/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 59000, episodes  59, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:27:53,417 | TRAIN LOG: steps 59000, episodes  59, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 59000, episodes  59, returns 2422.26/1553.68/816.67/5187.10/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:27:54,600 | TEST LOG: steps 59000, episodes  59, returns 2422.26/1553.68/816.67/5187.10/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 59000, episodes  59, returns 0.65/0.45/0.27/1.31/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 18:27:54,600 | Normalized LOG: steps 59000, episodes  59, returns 0.65/0.45/0.27/1.31/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 60000, episodes  60, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.03 steps/s
2024-03-28 18:28:39,985 | TRAIN LOG: steps 60000, episodes  60, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.03 steps/s
INFO:root:TEST LOG: steps 60000, episodes  60, returns 3213.30/4336.22/982.18/4987.78/5 (mean/median/min/max/num), 22.03 steps/s
2024-03-28 18:28:42,136 | TEST LOG: steps 60000, episodes  60, returns 3213.30/4336.22/982.18/4987.78/5 (mean/median/min/max/num), 22.03 steps/s
INFO:root:Normalized LOG: steps 60000, episodes  60, returns 0.84/1.11/0.31/1.26/5 (mean/median/min/max/num), 22.03 steps/s
2024-03-28 18:28:42,136 | Normalized LOG: steps 60000, episodes  60, returns 0.84/1.11/0.31/1.26/5 (mean/median/min/max/num), 22.03 steps/s
INFO:root:TRAIN LOG: steps 61000, episodes  61, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 18:29:27,565 | TRAIN LOG: steps 61000, episodes  61, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:TEST LOG: steps 61000, episodes  61, returns 3624.60/5028.25/-792.67/5314.78/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 18:29:29,930 | TEST LOG: steps 61000, episodes  61, returns 3624.60/5028.25/-792.67/5314.78/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:Normalized LOG: steps 61000, episodes  61, returns 0.94/1.27/-0.11/1.34/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 18:29:29,931 | Normalized LOG: steps 61000, episodes  61, returns 0.94/1.27/-0.11/1.34/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:TRAIN LOG: steps 62000, episodes  62, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.04 steps/s
2024-03-28 18:30:15,302 | TRAIN LOG: steps 62000, episodes  62, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.04 steps/s
INFO:root:TEST LOG: steps 62000, episodes  62, returns 3281.84/3646.94/647.98/5179.02/5 (mean/median/min/max/num), 22.04 steps/s
2024-03-28 18:30:17,057 | TEST LOG: steps 62000, episodes  62, returns 3281.84/3646.94/647.98/5179.02/5 (mean/median/min/max/num), 22.04 steps/s
INFO:root:Normalized LOG: steps 62000, episodes  62, returns 0.86/0.94/0.23/1.31/5 (mean/median/min/max/num), 22.04 steps/s
2024-03-28 18:30:17,057 | Normalized LOG: steps 62000, episodes  62, returns 0.86/0.94/0.23/1.31/5 (mean/median/min/max/num), 22.04 steps/s
INFO:root:TRAIN LOG: steps 63000, episodes  63, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:31:02,340 | TRAIN LOG: steps 63000, episodes  63, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TEST LOG: steps 63000, episodes  63, returns 5340.81/5289.45/5241.58/5585.08/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:31:04,831 | TEST LOG: steps 63000, episodes  63, returns 5340.81/5289.45/5241.58/5585.08/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:Normalized LOG: steps 63000, episodes  63, returns 1.35/1.34/1.32/1.41/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 18:31:04,831 | Normalized LOG: steps 63000, episodes  63, returns 1.35/1.34/1.32/1.41/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TRAIN LOG: steps 64000, episodes  64, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.96 steps/s
2024-03-28 18:31:50,360 | TRAIN LOG: steps 64000, episodes  64, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.96 steps/s
INFO:root:TEST LOG: steps 64000, episodes  64, returns 4178.01/5169.12/310.03/5369.96/5 (mean/median/min/max/num), 21.96 steps/s
2024-03-28 18:31:53,049 | TEST LOG: steps 64000, episodes  64, returns 4178.01/5169.12/310.03/5369.96/5 (mean/median/min/max/num), 21.96 steps/s
INFO:root:Normalized LOG: steps 64000, episodes  64, returns 1.07/1.31/0.15/1.35/5 (mean/median/min/max/num), 21.96 steps/s
2024-03-28 18:31:53,049 | Normalized LOG: steps 64000, episodes  64, returns 1.07/1.31/0.15/1.35/5 (mean/median/min/max/num), 21.96 steps/s
INFO:root:TRAIN LOG: steps 65000, episodes  65, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.95 steps/s
2024-03-28 18:32:38,615 | TRAIN LOG: steps 65000, episodes  65, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.95 steps/s
INFO:root:TEST LOG: steps 65000, episodes  65, returns 3033.86/4692.75/-1139.26/5094.51/5 (mean/median/min/max/num), 21.95 steps/s
2024-03-28 18:32:40,802 | TEST LOG: steps 65000, episodes  65, returns 3033.86/4692.75/-1139.26/5094.51/5 (mean/median/min/max/num), 21.95 steps/s
INFO:root:Normalized LOG: steps 65000, episodes  65, returns 0.80/1.19/-0.19/1.29/5 (mean/median/min/max/num), 21.95 steps/s
2024-03-28 18:32:40,803 | Normalized LOG: steps 65000, episodes  65, returns 0.80/1.19/-0.19/1.29/5 (mean/median/min/max/num), 21.95 steps/s
INFO:root:TRAIN LOG: steps 66000, episodes  66, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.73 steps/s
2024-03-28 18:33:26,820 | TRAIN LOG: steps 66000, episodes  66, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.73 steps/s
INFO:root:TEST LOG: steps 66000, episodes  66, returns 4806.62/4675.61/4316.37/5374.80/5 (mean/median/min/max/num), 21.73 steps/s
2024-03-28 18:33:29,241 | TEST LOG: steps 66000, episodes  66, returns 4806.62/4675.61/4316.37/5374.80/5 (mean/median/min/max/num), 21.73 steps/s
INFO:root:Normalized LOG: steps 66000, episodes  66, returns 1.22/1.19/1.10/1.36/5 (mean/median/min/max/num), 21.73 steps/s
2024-03-28 18:33:29,242 | Normalized LOG: steps 66000, episodes  66, returns 1.22/1.19/1.10/1.36/5 (mean/median/min/max/num), 21.73 steps/s
INFO:root:TRAIN LOG: steps 67000, episodes  67, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.64 steps/s
2024-03-28 18:34:15,451 | TRAIN LOG: steps 67000, episodes  67, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.64 steps/s
INFO:root:TEST LOG: steps 67000, episodes  67, returns 5148.60/5208.23/4770.07/5369.90/5 (mean/median/min/max/num), 21.64 steps/s
2024-03-28 18:34:17,944 | TEST LOG: steps 67000, episodes  67, returns 5148.60/5208.23/4770.07/5369.90/5 (mean/median/min/max/num), 21.64 steps/s
INFO:root:Normalized LOG: steps 67000, episodes  67, returns 1.30/1.32/1.21/1.35/5 (mean/median/min/max/num), 21.64 steps/s
2024-03-28 18:34:17,944 | Normalized LOG: steps 67000, episodes  67, returns 1.30/1.32/1.21/1.35/5 (mean/median/min/max/num), 21.64 steps/s
INFO:root:TRAIN LOG: steps 68000, episodes  68, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.50 steps/s
2024-03-28 18:35:04,464 | TRAIN LOG: steps 68000, episodes  68, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.50 steps/s
INFO:root:TEST LOG: steps 68000, episodes  68, returns 3008.99/3435.90/-280.60/5022.54/5 (mean/median/min/max/num), 21.50 steps/s
2024-03-28 18:35:07,070 | TEST LOG: steps 68000, episodes  68, returns 3008.99/3435.90/-280.60/5022.54/5 (mean/median/min/max/num), 21.50 steps/s
INFO:root:Normalized LOG: steps 68000, episodes  68, returns 0.79/0.89/0.01/1.27/5 (mean/median/min/max/num), 21.50 steps/s
2024-03-28 18:35:07,070 | Normalized LOG: steps 68000, episodes  68, returns 0.79/0.89/0.01/1.27/5 (mean/median/min/max/num), 21.50 steps/s
INFO:root:TRAIN LOG: steps 69000, episodes  69, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.36 steps/s
2024-03-28 18:35:53,878 | TRAIN LOG: steps 69000, episodes  69, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.36 steps/s
INFO:root:TEST LOG: steps 69000, episodes  69, returns 4482.97/4982.26/3322.47/5100.19/5 (mean/median/min/max/num), 21.36 steps/s
2024-03-28 18:35:56,097 | TEST LOG: steps 69000, episodes  69, returns 4482.97/4982.26/3322.47/5100.19/5 (mean/median/min/max/num), 21.36 steps/s
INFO:root:Normalized LOG: steps 69000, episodes  69, returns 1.14/1.26/0.87/1.29/5 (mean/median/min/max/num), 21.36 steps/s
2024-03-28 18:35:56,097 | Normalized LOG: steps 69000, episodes  69, returns 1.14/1.26/0.87/1.29/5 (mean/median/min/max/num), 21.36 steps/s
INFO:root:TRAIN LOG: steps 70000, episodes  70, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 18:36:43,610 | TRAIN LOG: steps 70000, episodes  70, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:TEST LOG: steps 70000, episodes  70, returns 3539.53/4596.17/-0.02/5321.68/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 18:36:46,011 | TEST LOG: steps 70000, episodes  70, returns 3539.53/4596.17/-0.02/5321.68/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:Normalized LOG: steps 70000, episodes  70, returns 0.92/1.17/0.08/1.34/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 18:36:46,011 | Normalized LOG: steps 70000, episodes  70, returns 0.92/1.17/0.08/1.34/5 (mean/median/min/max/num), 21.05 steps/s
"""

log_output = """INFO:root:TRAIN LOG: steps 0, episodes   0, returns 914.99/915.26/908.34/921.95/5 (mean/median/min/max/num), 359.79 steps/s
2024-03-28 19:00:25,851 | TRAIN LOG: steps 0, episodes   0, returns 914.99/915.26/908.34/921.95/5 (mean/median/min/max/num), 359.79 steps/s
INFO:root:TEST LOG: steps 0, episodes   0, returns 912.41/910.82/906.20/922.21/5 (mean/median/min/max/num), 359.79 steps/s
2024-03-28 19:00:28,635 | TEST LOG: steps 0, episodes   0, returns 912.41/910.82/906.20/922.21/5 (mean/median/min/max/num), 359.79 steps/s
INFO:root:Normalized LOG: steps 0, episodes   0, returns 0.29/0.29/0.29/0.30/5 (mean/median/min/max/num), 359.79 steps/s
2024-03-28 19:00:28,635 | Normalized LOG: steps 0, episodes   0, returns 0.29/0.29/0.29/0.30/5 (mean/median/min/max/num), 359.79 steps/s
INFO:root:TRAIN LOG: steps 1000, episodes   1, returns 730.95/909.22/0.00/921.95/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 19:01:13,859 | TRAIN LOG: steps 1000, episodes   1, returns 730.95/909.22/0.00/921.95/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TEST LOG: steps 1000, episodes   1, returns 500.94/577.88/205.51/598.75/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 19:01:16,627 | TEST LOG: steps 1000, episodes   1, returns 500.94/577.88/205.51/598.75/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:Normalized LOG: steps 1000, episodes   1, returns 0.20/0.21/0.13/0.22/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 19:01:16,627 | Normalized LOG: steps 1000, episodes   1, returns 0.20/0.21/0.13/0.22/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TRAIN LOG: steps 2000, episodes   2, returns 549.11/908.34/0.00/921.95/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:02:01,891 | TRAIN LOG: steps 2000, episodes   2, returns 549.11/908.34/0.00/921.95/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TEST LOG: steps 2000, episodes   2, returns 1286.37/1023.81/-123.94/3669.86/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:02:04,135 | TEST LOG: steps 2000, episodes   2, returns 1286.37/1023.81/-123.94/3669.86/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:Normalized LOG: steps 2000, episodes   2, returns 0.38/0.32/0.05/0.95/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:02:04,135 | Normalized LOG: steps 2000, episodes   2, returns 0.38/0.32/0.05/0.95/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TRAIN LOG: steps 3000, episodes   3, returns 366.06/0.00/0.00/921.95/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:02:49,645 | TRAIN LOG: steps 3000, episodes   3, returns 366.06/0.00/0.00/921.95/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TEST LOG: steps 3000, episodes   3, returns 2063.17/2952.74/-487.60/3870.64/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:02:52,255 | TEST LOG: steps 3000, episodes   3, returns 2063.17/2952.74/-487.60/3870.64/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:Normalized LOG: steps 3000, episodes   3, returns 0.57/0.78/-0.04/1.00/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:02:52,255 | Normalized LOG: steps 3000, episodes   3, returns 0.57/0.78/-0.04/1.00/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TRAIN LOG: steps 4000, episodes   4, returns 181.67/0.00/0.00/908.34/5 (mean/median/min/max/num), 21.93 steps/s
2024-03-28 19:03:37,856 | TRAIN LOG: steps 4000, episodes   4, returns 181.67/0.00/0.00/908.34/5 (mean/median/min/max/num), 21.93 steps/s
INFO:root:TEST LOG: steps 4000, episodes   4, returns 1529.61/1492.05/358.82/2841.17/5 (mean/median/min/max/num), 21.93 steps/s
2024-03-28 19:03:40,049 | TEST LOG: steps 4000, episodes   4, returns 1529.61/1492.05/358.82/2841.17/5 (mean/median/min/max/num), 21.93 steps/s
INFO:root:Normalized LOG: steps 4000, episodes   4, returns 0.44/0.43/0.16/0.75/5 (mean/median/min/max/num), 21.93 steps/s
2024-03-28 19:03:40,050 | Normalized LOG: steps 4000, episodes   4, returns 0.44/0.43/0.16/0.75/5 (mean/median/min/max/num), 21.93 steps/s
INFO:root:TRAIN LOG: steps 5000, episodes   5, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.00 steps/s
2024-03-28 19:04:25,502 | TRAIN LOG: steps 5000, episodes   5, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.00 steps/s
INFO:root:TEST LOG: steps 5000, episodes   5, returns 3393.17/3968.04/539.17/4623.98/5 (mean/median/min/max/num), 22.00 steps/s
2024-03-28 19:04:27,467 | TEST LOG: steps 5000, episodes   5, returns 3393.17/3968.04/539.17/4623.98/5 (mean/median/min/max/num), 22.00 steps/s
INFO:root:Normalized LOG: steps 5000, episodes   5, returns 0.88/1.02/0.21/1.18/5 (mean/median/min/max/num), 22.00 steps/s
2024-03-28 19:04:27,467 | Normalized LOG: steps 5000, episodes   5, returns 0.88/1.02/0.21/1.18/5 (mean/median/min/max/num), 22.00 steps/s
INFO:root:TRAIN LOG: steps 6000, episodes   6, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:05:12,992 | TRAIN LOG: steps 6000, episodes   6, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TEST LOG: steps 6000, episodes   6, returns 2794.75/2555.24/480.31/4953.23/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:05:14,578 | TEST LOG: steps 6000, episodes   6, returns 2794.75/2555.24/480.31/4953.23/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:Normalized LOG: steps 6000, episodes   6, returns 0.74/0.69/0.19/1.26/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:05:14,579 | Normalized LOG: steps 6000, episodes   6, returns 0.74/0.69/0.19/1.26/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TRAIN LOG: steps 7000, episodes   7, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 19:05:59,773 | TRAIN LOG: steps 7000, episodes   7, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 7000, episodes   7, returns 620.45/472.76/288.39/1560.59/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 19:06:01,123 | TEST LOG: steps 7000, episodes   7, returns 620.45/472.76/288.39/1560.59/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 7000, episodes   7, returns 0.22/0.19/0.15/0.45/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 19:06:01,123 | Normalized LOG: steps 7000, episodes   7, returns 0.22/0.19/0.15/0.45/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 8000, episodes   8, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:06:46,648 | TRAIN LOG: steps 8000, episodes   8, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TEST LOG: steps 8000, episodes   8, returns 4569.21/4421.14/4124.46/5198.42/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:06:49,148 | TEST LOG: steps 8000, episodes   8, returns 4569.21/4421.14/4124.46/5198.42/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:Normalized LOG: steps 8000, episodes   8, returns 1.16/1.13/1.06/1.31/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:06:49,148 | Normalized LOG: steps 8000, episodes   8, returns 1.16/1.13/1.06/1.31/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TRAIN LOG: steps 9000, episodes   9, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:07:34,200 | TRAIN LOG: steps 9000, episodes   9, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TEST LOG: steps 9000, episodes   9, returns 3881.59/4679.52/873.59/4865.10/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:07:36,294 | TEST LOG: steps 9000, episodes   9, returns 3881.59/4679.52/873.59/4865.10/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:Normalized LOG: steps 9000, episodes   9, returns 1.00/1.19/0.29/1.23/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:07:36,294 | Normalized LOG: steps 9000, episodes   9, returns 1.00/1.19/0.29/1.23/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TRAIN LOG: steps 10000, episodes  10, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 19:08:21,737 | TRAIN LOG: steps 10000, episodes  10, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:TEST LOG: steps 10000, episodes  10, returns 2625.36/2987.12/933.27/4704.32/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 19:08:23,266 | TEST LOG: steps 10000, episodes  10, returns 2625.36/2987.12/933.27/4704.32/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:Normalized LOG: steps 10000, episodes  10, returns 0.70/0.79/0.30/1.20/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 19:08:23,266 | Normalized LOG: steps 10000, episodes  10, returns 0.70/0.79/0.30/1.20/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:TRAIN LOG: steps 11000, episodes  11, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:09:09,982 | TRAIN LOG: steps 11000, episodes  11, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TEST LOG: steps 11000, episodes  11, returns 1838.22/1998.81/122.02/4510.29/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:09:11,868 | TEST LOG: steps 11000, episodes  11, returns 1838.22/1998.81/122.02/4510.29/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:Normalized LOG: steps 11000, episodes  11, returns 0.51/0.55/0.11/1.15/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:09:11,868 | Normalized LOG: steps 11000, episodes  11, returns 0.51/0.55/0.11/1.15/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TRAIN LOG: steps 12000, episodes  12, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.45 steps/s
2024-03-28 19:09:58,488 | TRAIN LOG: steps 12000, episodes  12, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.45 steps/s
INFO:root:TEST LOG: steps 12000, episodes  12, returns 4162.91/4828.39/2561.62/5095.30/5 (mean/median/min/max/num), 21.45 steps/s
2024-03-28 19:10:01,086 | TEST LOG: steps 12000, episodes  12, returns 4162.91/4828.39/2561.62/5095.30/5 (mean/median/min/max/num), 21.45 steps/s
INFO:root:Normalized LOG: steps 12000, episodes  12, returns 1.07/1.23/0.69/1.29/5 (mean/median/min/max/num), 21.45 steps/s
2024-03-28 19:10:01,086 | Normalized LOG: steps 12000, episodes  12, returns 1.07/1.23/0.69/1.29/5 (mean/median/min/max/num), 21.45 steps/s
INFO:root:TRAIN LOG: steps 13000, episodes  13, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.80 steps/s
2024-03-28 19:10:46,963 | TRAIN LOG: steps 13000, episodes  13, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.80 steps/s
INFO:root:TEST LOG: steps 13000, episodes  13, returns 2684.68/2753.81/128.65/4778.27/5 (mean/median/min/max/num), 21.80 steps/s
2024-03-28 19:10:48,736 | TEST LOG: steps 13000, episodes  13, returns 2684.68/2753.81/128.65/4778.27/5 (mean/median/min/max/num), 21.80 steps/s
INFO:root:Normalized LOG: steps 13000, episodes  13, returns 0.72/0.73/0.11/1.21/5 (mean/median/min/max/num), 21.80 steps/s
2024-03-28 19:10:48,736 | Normalized LOG: steps 13000, episodes  13, returns 0.72/0.73/0.11/1.21/5 (mean/median/min/max/num), 21.80 steps/s
INFO:root:TRAIN LOG: steps 14000, episodes  14, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 19:11:34,024 | TRAIN LOG: steps 14000, episodes  14, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TEST LOG: steps 14000, episodes  14, returns 3592.80/4004.96/1081.17/4657.29/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 19:11:36,020 | TEST LOG: steps 14000, episodes  14, returns 3592.80/4004.96/1081.17/4657.29/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:Normalized LOG: steps 14000, episodes  14, returns 0.93/1.03/0.33/1.18/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 19:11:36,020 | Normalized LOG: steps 14000, episodes  14, returns 0.93/1.03/0.33/1.18/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TRAIN LOG: steps 15000, episodes  15, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:12:22,722 | TRAIN LOG: steps 15000, episodes  15, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TEST LOG: steps 15000, episodes  15, returns 4387.09/4879.36/2473.70/5081.29/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:12:24,983 | TEST LOG: steps 15000, episodes  15, returns 4387.09/4879.36/2473.70/5081.29/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:Normalized LOG: steps 15000, episodes  15, returns 1.12/1.24/0.67/1.29/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:12:24,983 | Normalized LOG: steps 15000, episodes  15, returns 1.12/1.24/0.67/1.29/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TRAIN LOG: steps 16000, episodes  16, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.76 steps/s
2024-03-28 19:13:10,947 | TRAIN LOG: steps 16000, episodes  16, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.76 steps/s
INFO:root:TEST LOG: steps 16000, episodes  16, returns 2775.00/2499.78/294.19/4667.56/5 (mean/median/min/max/num), 21.76 steps/s
2024-03-28 19:13:13,339 | TEST LOG: steps 16000, episodes  16, returns 2775.00/2499.78/294.19/4667.56/5 (mean/median/min/max/num), 21.76 steps/s
INFO:root:Normalized LOG: steps 16000, episodes  16, returns 0.74/0.67/0.15/1.19/5 (mean/median/min/max/num), 21.76 steps/s
2024-03-28 19:13:13,339 | Normalized LOG: steps 16000, episodes  16, returns 0.74/0.67/0.15/1.19/5 (mean/median/min/max/num), 21.76 steps/s
INFO:root:TRAIN LOG: steps 17000, episodes  17, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:13:59,508 | TRAIN LOG: steps 17000, episodes  17, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TEST LOG: steps 17000, episodes  17, returns 2185.75/1451.20/427.87/4610.67/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:14:00,862 | TEST LOG: steps 17000, episodes  17, returns 2185.75/1451.20/427.87/4610.67/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:Normalized LOG: steps 17000, episodes  17, returns 0.60/0.42/0.18/1.17/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:14:00,862 | Normalized LOG: steps 17000, episodes  17, returns 0.60/0.42/0.18/1.17/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TRAIN LOG: steps 18000, episodes  18, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.02 steps/s
2024-03-28 19:14:46,281 | TRAIN LOG: steps 18000, episodes  18, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.02 steps/s
INFO:root:TEST LOG: steps 18000, episodes  18, returns 3829.01/4519.97/781.96/5305.69/5 (mean/median/min/max/num), 22.02 steps/s
2024-03-28 19:14:48,266 | TEST LOG: steps 18000, episodes  18, returns 3829.01/4519.97/781.96/5305.69/5 (mean/median/min/max/num), 22.02 steps/s
INFO:root:Normalized LOG: steps 18000, episodes  18, returns 0.99/1.15/0.26/1.34/5 (mean/median/min/max/num), 22.02 steps/s
2024-03-28 19:14:48,266 | Normalized LOG: steps 18000, episodes  18, returns 0.99/1.15/0.26/1.34/5 (mean/median/min/max/num), 22.02 steps/s
INFO:root:TRAIN LOG: steps 19000, episodes  19, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.77 steps/s
2024-03-28 19:15:34,196 | TRAIN LOG: steps 19000, episodes  19, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.77 steps/s
INFO:root:TEST LOG: steps 19000, episodes  19, returns 3086.91/2713.62/950.43/5400.04/5 (mean/median/min/max/num), 21.77 steps/s
2024-03-28 19:15:35,738 | TEST LOG: steps 19000, episodes  19, returns 3086.91/2713.62/950.43/5400.04/5 (mean/median/min/max/num), 21.77 steps/s
INFO:root:Normalized LOG: steps 19000, episodes  19, returns 0.81/0.72/0.30/1.36/5 (mean/median/min/max/num), 21.77 steps/s
2024-03-28 19:15:35,738 | Normalized LOG: steps 19000, episodes  19, returns 0.81/0.72/0.30/1.36/5 (mean/median/min/max/num), 21.77 steps/s
INFO:root:TRAIN LOG: steps 20000, episodes  20, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:16:21,011 | TRAIN LOG: steps 20000, episodes  20, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TEST LOG: steps 20000, episodes  20, returns 2239.50/1429.01/157.78/4755.45/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:16:22,195 | TEST LOG: steps 20000, episodes  20, returns 2239.50/1429.01/157.78/4755.45/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:Normalized LOG: steps 20000, episodes  20, returns 0.61/0.42/0.11/1.21/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:16:22,196 | Normalized LOG: steps 20000, episodes  20, returns 0.61/0.42/0.11/1.21/5 (mean/median/min/max/num), 22.09 steps/s
"""

log_output = """INFO:root:TRAIN LOG: steps 0, episodes   0, returns 914.99/915.26/908.34/921.95/5 (mean/median/min/max/num), 359.79 steps/s
2024-03-28 19:00:25,851 | TRAIN LOG: steps 0, episodes   0, returns 914.99/915.26/908.34/921.95/5 (mean/median/min/max/num), 359.79 steps/s
INFO:root:TEST LOG: steps 0, episodes   0, returns 912.41/910.82/906.20/922.21/5 (mean/median/min/max/num), 359.79 steps/s
2024-03-28 19:00:28,635 | TEST LOG: steps 0, episodes   0, returns 912.41/910.82/906.20/922.21/5 (mean/median/min/max/num), 359.79 steps/s
INFO:root:Normalized LOG: steps 0, episodes   0, returns 0.29/0.29/0.29/0.30/5 (mean/median/min/max/num), 359.79 steps/s
2024-03-28 19:00:28,635 | Normalized LOG: steps 0, episodes   0, returns 0.29/0.29/0.29/0.30/5 (mean/median/min/max/num), 359.79 steps/s
INFO:root:TRAIN LOG: steps 1000, episodes   1, returns 730.95/909.22/0.00/921.95/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 19:01:13,859 | TRAIN LOG: steps 1000, episodes   1, returns 730.95/909.22/0.00/921.95/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TEST LOG: steps 1000, episodes   1, returns 500.94/577.88/205.51/598.75/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 19:01:16,627 | TEST LOG: steps 1000, episodes   1, returns 500.94/577.88/205.51/598.75/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:Normalized LOG: steps 1000, episodes   1, returns 0.20/0.21/0.13/0.22/5 (mean/median/min/max/num), 22.11 steps/s
2024-03-28 19:01:16,627 | Normalized LOG: steps 1000, episodes   1, returns 0.20/0.21/0.13/0.22/5 (mean/median/min/max/num), 22.11 steps/s
INFO:root:TRAIN LOG: steps 2000, episodes   2, returns 549.11/908.34/0.00/921.95/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:02:01,891 | TRAIN LOG: steps 2000, episodes   2, returns 549.11/908.34/0.00/921.95/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TEST LOG: steps 2000, episodes   2, returns 1286.37/1023.81/-123.94/3669.86/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:02:04,135 | TEST LOG: steps 2000, episodes   2, returns 1286.37/1023.81/-123.94/3669.86/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:Normalized LOG: steps 2000, episodes   2, returns 0.38/0.32/0.05/0.95/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:02:04,135 | Normalized LOG: steps 2000, episodes   2, returns 0.38/0.32/0.05/0.95/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TRAIN LOG: steps 3000, episodes   3, returns 366.06/0.00/0.00/921.95/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:02:49,645 | TRAIN LOG: steps 3000, episodes   3, returns 366.06/0.00/0.00/921.95/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TEST LOG: steps 3000, episodes   3, returns 2063.17/2952.74/-487.60/3870.64/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:02:52,255 | TEST LOG: steps 3000, episodes   3, returns 2063.17/2952.74/-487.60/3870.64/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:Normalized LOG: steps 3000, episodes   3, returns 0.57/0.78/-0.04/1.00/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:02:52,255 | Normalized LOG: steps 3000, episodes   3, returns 0.57/0.78/-0.04/1.00/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TRAIN LOG: steps 4000, episodes   4, returns 181.67/0.00/0.00/908.34/5 (mean/median/min/max/num), 21.93 steps/s
2024-03-28 19:03:37,856 | TRAIN LOG: steps 4000, episodes   4, returns 181.67/0.00/0.00/908.34/5 (mean/median/min/max/num), 21.93 steps/s
INFO:root:TEST LOG: steps 4000, episodes   4, returns 1529.61/1492.05/358.82/2841.17/5 (mean/median/min/max/num), 21.93 steps/s
2024-03-28 19:03:40,049 | TEST LOG: steps 4000, episodes   4, returns 1529.61/1492.05/358.82/2841.17/5 (mean/median/min/max/num), 21.93 steps/s
INFO:root:Normalized LOG: steps 4000, episodes   4, returns 0.44/0.43/0.16/0.75/5 (mean/median/min/max/num), 21.93 steps/s
2024-03-28 19:03:40,050 | Normalized LOG: steps 4000, episodes   4, returns 0.44/0.43/0.16/0.75/5 (mean/median/min/max/num), 21.93 steps/s
INFO:root:TRAIN LOG: steps 5000, episodes   5, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.00 steps/s
2024-03-28 19:04:25,502 | TRAIN LOG: steps 5000, episodes   5, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.00 steps/s
INFO:root:TEST LOG: steps 5000, episodes   5, returns 3393.17/3968.04/539.17/4623.98/5 (mean/median/min/max/num), 22.00 steps/s
2024-03-28 19:04:27,467 | TEST LOG: steps 5000, episodes   5, returns 3393.17/3968.04/539.17/4623.98/5 (mean/median/min/max/num), 22.00 steps/s
INFO:root:Normalized LOG: steps 5000, episodes   5, returns 0.88/1.02/0.21/1.18/5 (mean/median/min/max/num), 22.00 steps/s
2024-03-28 19:04:27,467 | Normalized LOG: steps 5000, episodes   5, returns 0.88/1.02/0.21/1.18/5 (mean/median/min/max/num), 22.00 steps/s
INFO:root:TRAIN LOG: steps 6000, episodes   6, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:05:12,992 | TRAIN LOG: steps 6000, episodes   6, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TEST LOG: steps 6000, episodes   6, returns 2794.75/2555.24/480.31/4953.23/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:05:14,578 | TEST LOG: steps 6000, episodes   6, returns 2794.75/2555.24/480.31/4953.23/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:Normalized LOG: steps 6000, episodes   6, returns 0.74/0.69/0.19/1.26/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:05:14,579 | Normalized LOG: steps 6000, episodes   6, returns 0.74/0.69/0.19/1.26/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TRAIN LOG: steps 7000, episodes   7, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 19:05:59,773 | TRAIN LOG: steps 7000, episodes   7, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TEST LOG: steps 7000, episodes   7, returns 620.45/472.76/288.39/1560.59/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 19:06:01,123 | TEST LOG: steps 7000, episodes   7, returns 620.45/472.76/288.39/1560.59/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:Normalized LOG: steps 7000, episodes   7, returns 0.22/0.19/0.15/0.45/5 (mean/median/min/max/num), 22.13 steps/s
2024-03-28 19:06:01,123 | Normalized LOG: steps 7000, episodes   7, returns 0.22/0.19/0.15/0.45/5 (mean/median/min/max/num), 22.13 steps/s
INFO:root:TRAIN LOG: steps 8000, episodes   8, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:06:46,648 | TRAIN LOG: steps 8000, episodes   8, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TEST LOG: steps 8000, episodes   8, returns 4569.21/4421.14/4124.46/5198.42/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:06:49,148 | TEST LOG: steps 8000, episodes   8, returns 4569.21/4421.14/4124.46/5198.42/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:Normalized LOG: steps 8000, episodes   8, returns 1.16/1.13/1.06/1.31/5 (mean/median/min/max/num), 21.97 steps/s
2024-03-28 19:06:49,148 | Normalized LOG: steps 8000, episodes   8, returns 1.16/1.13/1.06/1.31/5 (mean/median/min/max/num), 21.97 steps/s
INFO:root:TRAIN LOG: steps 9000, episodes   9, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:07:34,200 | TRAIN LOG: steps 9000, episodes   9, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TEST LOG: steps 9000, episodes   9, returns 3881.59/4679.52/873.59/4865.10/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:07:36,294 | TEST LOG: steps 9000, episodes   9, returns 3881.59/4679.52/873.59/4865.10/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:Normalized LOG: steps 9000, episodes   9, returns 1.00/1.19/0.29/1.23/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:07:36,294 | Normalized LOG: steps 9000, episodes   9, returns 1.00/1.19/0.29/1.23/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TRAIN LOG: steps 10000, episodes  10, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 19:08:21,737 | TRAIN LOG: steps 10000, episodes  10, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:TEST LOG: steps 10000, episodes  10, returns 2625.36/2987.12/933.27/4704.32/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 19:08:23,266 | TEST LOG: steps 10000, episodes  10, returns 2625.36/2987.12/933.27/4704.32/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:Normalized LOG: steps 10000, episodes  10, returns 0.70/0.79/0.30/1.20/5 (mean/median/min/max/num), 22.01 steps/s
2024-03-28 19:08:23,266 | Normalized LOG: steps 10000, episodes  10, returns 0.70/0.79/0.30/1.20/5 (mean/median/min/max/num), 22.01 steps/s
INFO:root:TRAIN LOG: steps 11000, episodes  11, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:09:09,982 | TRAIN LOG: steps 11000, episodes  11, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TEST LOG: steps 11000, episodes  11, returns 1838.22/1998.81/122.02/4510.29/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:09:11,868 | TEST LOG: steps 11000, episodes  11, returns 1838.22/1998.81/122.02/4510.29/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:Normalized LOG: steps 11000, episodes  11, returns 0.51/0.55/0.11/1.15/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:09:11,868 | Normalized LOG: steps 11000, episodes  11, returns 0.51/0.55/0.11/1.15/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TRAIN LOG: steps 12000, episodes  12, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.45 steps/s
2024-03-28 19:09:58,488 | TRAIN LOG: steps 12000, episodes  12, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.45 steps/s
INFO:root:TEST LOG: steps 12000, episodes  12, returns 4162.91/4828.39/2561.62/5095.30/5 (mean/median/min/max/num), 21.45 steps/s
2024-03-28 19:10:01,086 | TEST LOG: steps 12000, episodes  12, returns 4162.91/4828.39/2561.62/5095.30/5 (mean/median/min/max/num), 21.45 steps/s
INFO:root:Normalized LOG: steps 12000, episodes  12, returns 1.07/1.23/0.69/1.29/5 (mean/median/min/max/num), 21.45 steps/s
2024-03-28 19:10:01,086 | Normalized LOG: steps 12000, episodes  12, returns 1.07/1.23/0.69/1.29/5 (mean/median/min/max/num), 21.45 steps/s
INFO:root:TRAIN LOG: steps 13000, episodes  13, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.80 steps/s
2024-03-28 19:10:46,963 | TRAIN LOG: steps 13000, episodes  13, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.80 steps/s
INFO:root:TEST LOG: steps 13000, episodes  13, returns 2684.68/2753.81/128.65/4778.27/5 (mean/median/min/max/num), 21.80 steps/s
2024-03-28 19:10:48,736 | TEST LOG: steps 13000, episodes  13, returns 2684.68/2753.81/128.65/4778.27/5 (mean/median/min/max/num), 21.80 steps/s
INFO:root:Normalized LOG: steps 13000, episodes  13, returns 0.72/0.73/0.11/1.21/5 (mean/median/min/max/num), 21.80 steps/s
2024-03-28 19:10:48,736 | Normalized LOG: steps 13000, episodes  13, returns 0.72/0.73/0.11/1.21/5 (mean/median/min/max/num), 21.80 steps/s
INFO:root:TRAIN LOG: steps 14000, episodes  14, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 19:11:34,024 | TRAIN LOG: steps 14000, episodes  14, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TEST LOG: steps 14000, episodes  14, returns 3592.80/4004.96/1081.17/4657.29/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 19:11:36,020 | TEST LOG: steps 14000, episodes  14, returns 3592.80/4004.96/1081.17/4657.29/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:Normalized LOG: steps 14000, episodes  14, returns 0.93/1.03/0.33/1.18/5 (mean/median/min/max/num), 22.08 steps/s
2024-03-28 19:11:36,020 | Normalized LOG: steps 14000, episodes  14, returns 0.93/1.03/0.33/1.18/5 (mean/median/min/max/num), 22.08 steps/s
INFO:root:TRAIN LOG: steps 15000, episodes  15, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:12:22,722 | TRAIN LOG: steps 15000, episodes  15, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TEST LOG: steps 15000, episodes  15, returns 4387.09/4879.36/2473.70/5081.29/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:12:24,983 | TEST LOG: steps 15000, episodes  15, returns 4387.09/4879.36/2473.70/5081.29/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:Normalized LOG: steps 15000, episodes  15, returns 1.12/1.24/0.67/1.29/5 (mean/median/min/max/num), 21.41 steps/s
2024-03-28 19:12:24,983 | Normalized LOG: steps 15000, episodes  15, returns 1.12/1.24/0.67/1.29/5 (mean/median/min/max/num), 21.41 steps/s
INFO:root:TRAIN LOG: steps 16000, episodes  16, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.76 steps/s
2024-03-28 19:13:10,947 | TRAIN LOG: steps 16000, episodes  16, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.76 steps/s
INFO:root:TEST LOG: steps 16000, episodes  16, returns 2775.00/2499.78/294.19/4667.56/5 (mean/median/min/max/num), 21.76 steps/s
2024-03-28 19:13:13,339 | TEST LOG: steps 16000, episodes  16, returns 2775.00/2499.78/294.19/4667.56/5 (mean/median/min/max/num), 21.76 steps/s
INFO:root:Normalized LOG: steps 16000, episodes  16, returns 0.74/0.67/0.15/1.19/5 (mean/median/min/max/num), 21.76 steps/s
2024-03-28 19:13:13,339 | Normalized LOG: steps 16000, episodes  16, returns 0.74/0.67/0.15/1.19/5 (mean/median/min/max/num), 21.76 steps/s
INFO:root:TRAIN LOG: steps 17000, episodes  17, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:13:59,508 | TRAIN LOG: steps 17000, episodes  17, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TEST LOG: steps 17000, episodes  17, returns 2185.75/1451.20/427.87/4610.67/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:14:00,862 | TEST LOG: steps 17000, episodes  17, returns 2185.75/1451.20/427.87/4610.67/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:Normalized LOG: steps 17000, episodes  17, returns 0.60/0.42/0.18/1.17/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:14:00,862 | Normalized LOG: steps 17000, episodes  17, returns 0.60/0.42/0.18/1.17/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TRAIN LOG: steps 18000, episodes  18, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.02 steps/s
2024-03-28 19:14:46,281 | TRAIN LOG: steps 18000, episodes  18, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.02 steps/s
INFO:root:TEST LOG: steps 18000, episodes  18, returns 3829.01/4519.97/781.96/5305.69/5 (mean/median/min/max/num), 22.02 steps/s
2024-03-28 19:14:48,266 | TEST LOG: steps 18000, episodes  18, returns 3829.01/4519.97/781.96/5305.69/5 (mean/median/min/max/num), 22.02 steps/s
INFO:root:Normalized LOG: steps 18000, episodes  18, returns 0.99/1.15/0.26/1.34/5 (mean/median/min/max/num), 22.02 steps/s
2024-03-28 19:14:48,266 | Normalized LOG: steps 18000, episodes  18, returns 0.99/1.15/0.26/1.34/5 (mean/median/min/max/num), 22.02 steps/s
INFO:root:TRAIN LOG: steps 19000, episodes  19, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.77 steps/s
2024-03-28 19:15:34,196 | TRAIN LOG: steps 19000, episodes  19, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.77 steps/s
INFO:root:TEST LOG: steps 19000, episodes  19, returns 3086.91/2713.62/950.43/5400.04/5 (mean/median/min/max/num), 21.77 steps/s
2024-03-28 19:15:35,738 | TEST LOG: steps 19000, episodes  19, returns 3086.91/2713.62/950.43/5400.04/5 (mean/median/min/max/num), 21.77 steps/s
INFO:root:Normalized LOG: steps 19000, episodes  19, returns 0.81/0.72/0.30/1.36/5 (mean/median/min/max/num), 21.77 steps/s
2024-03-28 19:15:35,738 | Normalized LOG: steps 19000, episodes  19, returns 0.81/0.72/0.30/1.36/5 (mean/median/min/max/num), 21.77 steps/s
INFO:root:TRAIN LOG: steps 20000, episodes  20, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:16:21,011 | TRAIN LOG: steps 20000, episodes  20, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TEST LOG: steps 20000, episodes  20, returns 2239.50/1429.01/157.78/4755.45/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:16:22,195 | TEST LOG: steps 20000, episodes  20, returns 2239.50/1429.01/157.78/4755.45/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:Normalized LOG: steps 20000, episodes  20, returns 0.61/0.42/0.11/1.21/5 (mean/median/min/max/num), 22.09 steps/s
2024-03-28 19:16:22,196 | Normalized LOG: steps 20000, episodes  20, returns 0.61/0.42/0.11/1.21/5 (mean/median/min/max/num), 22.09 steps/s
INFO:root:TRAIN LOG: steps 21000, episodes  21, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.89 steps/s
2024-03-28 19:17:07,872 | TRAIN LOG: steps 21000, episodes  21, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.89 steps/s
INFO:root:TEST LOG: steps 21000, episodes  21, returns 2484.68/2394.80/976.77/4955.40/5 (mean/median/min/max/num), 21.89 steps/s
2024-03-28 19:17:09,152 | TEST LOG: steps 21000, episodes  21, returns 2484.68/2394.80/976.77/4955.40/5 (mean/median/min/max/num), 21.89 steps/s
INFO:root:Normalized LOG: steps 21000, episodes  21, returns 0.67/0.65/0.31/1.26/5 (mean/median/min/max/num), 21.89 steps/s
2024-03-28 19:17:09,152 | Normalized LOG: steps 21000, episodes  21, returns 0.67/0.65/0.31/1.26/5 (mean/median/min/max/num), 21.89 steps/s
INFO:root:TRAIN LOG: steps 22000, episodes  22, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:17:55,330 | TRAIN LOG: steps 22000, episodes  22, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TEST LOG: steps 22000, episodes  22, returns 2694.64/4057.33/103.76/4474.51/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:17:56,644 | TEST LOG: steps 22000, episodes  22, returns 2694.64/4057.33/103.76/4474.51/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:Normalized LOG: steps 22000, episodes  22, returns 0.72/1.04/0.10/1.14/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:17:56,644 | Normalized LOG: steps 22000, episodes  22, returns 0.72/1.04/0.10/1.14/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TRAIN LOG: steps 23000, episodes  23, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.82 steps/s
2024-03-28 19:18:42,478 | TRAIN LOG: steps 23000, episodes  23, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.82 steps/s
INFO:root:TEST LOG: steps 23000, episodes  23, returns 2420.98/2795.57/-9.81/4854.72/5 (mean/median/min/max/num), 21.82 steps/s
2024-03-28 19:18:44,254 | TEST LOG: steps 23000, episodes  23, returns 2420.98/2795.57/-9.81/4854.72/5 (mean/median/min/max/num), 21.82 steps/s
INFO:root:Normalized LOG: steps 23000, episodes  23, returns 0.65/0.74/0.08/1.23/5 (mean/median/min/max/num), 21.82 steps/s
2024-03-28 19:18:44,254 | Normalized LOG: steps 23000, episodes  23, returns 0.65/0.74/0.08/1.23/5 (mean/median/min/max/num), 21.82 steps/s
INFO:root:TRAIN LOG: steps 24000, episodes  24, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.89 steps/s
2024-03-28 19:19:29,940 | TRAIN LOG: steps 24000, episodes  24, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.89 steps/s
INFO:root:TEST LOG: steps 24000, episodes  24, returns 4328.45/4161.35/3482.21/5144.45/5 (mean/median/min/max/num), 21.89 steps/s
2024-03-28 19:19:32,147 | TEST LOG: steps 24000, episodes  24, returns 4328.45/4161.35/3482.21/5144.45/5 (mean/median/min/max/num), 21.89 steps/s
INFO:root:Normalized LOG: steps 24000, episodes  24, returns 1.11/1.07/0.91/1.30/5 (mean/median/min/max/num), 21.89 steps/s
2024-03-28 19:19:32,147 | Normalized LOG: steps 24000, episodes  24, returns 1.11/1.07/0.91/1.30/5 (mean/median/min/max/num), 21.89 steps/s
INFO:root:TRAIN LOG: steps 25000, episodes  25, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 19:20:17,389 | TRAIN LOG: steps 25000, episodes  25, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:TEST LOG: steps 25000, episodes  25, returns 3716.93/5170.49/500.71/5460.31/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 19:20:19,973 | TEST LOG: steps 25000, episodes  25, returns 3716.93/5170.49/500.71/5460.31/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:Normalized LOG: steps 25000, episodes  25, returns 0.96/1.31/0.20/1.38/5 (mean/median/min/max/num), 22.10 steps/s
2024-03-28 19:20:19,973 | Normalized LOG: steps 25000, episodes  25, returns 0.96/1.31/0.20/1.38/5 (mean/median/min/max/num), 22.10 steps/s
INFO:root:TRAIN LOG: steps 26000, episodes  26, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.87 steps/s
2024-03-28 19:21:05,695 | TRAIN LOG: steps 26000, episodes  26, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.87 steps/s
INFO:root:TEST LOG: steps 26000, episodes  26, returns 4748.11/5135.23/3173.86/5199.60/5 (mean/median/min/max/num), 21.87 steps/s
2024-03-28 19:21:08,050 | TEST LOG: steps 26000, episodes  26, returns 4748.11/5135.23/3173.86/5199.60/5 (mean/median/min/max/num), 21.87 steps/s
INFO:root:Normalized LOG: steps 26000, episodes  26, returns 1.21/1.30/0.83/1.31/5 (mean/median/min/max/num), 21.87 steps/s
2024-03-28 19:21:08,051 | Normalized LOG: steps 26000, episodes  26, returns 1.21/1.30/0.83/1.31/5 (mean/median/min/max/num), 21.87 steps/s
INFO:root:TRAIN LOG: steps 27000, episodes  27, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:21:54,213 | TRAIN LOG: steps 27000, episodes  27, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TEST LOG: steps 27000, episodes  27, returns 2721.92/2540.14/1313.91/4421.01/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:21:55,652 | TEST LOG: steps 27000, episodes  27, returns 2721.92/2540.14/1313.91/4421.01/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:Normalized LOG: steps 27000, episodes  27, returns 0.72/0.68/0.39/1.13/5 (mean/median/min/max/num), 21.66 steps/s
2024-03-28 19:21:55,653 | Normalized LOG: steps 27000, episodes  27, returns 0.72/0.68/0.39/1.13/5 (mean/median/min/max/num), 21.66 steps/s
INFO:root:TRAIN LOG: steps 28000, episodes  28, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.86 steps/s
2024-03-28 19:22:41,401 | TRAIN LOG: steps 28000, episodes  28, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.86 steps/s
INFO:root:TEST LOG: steps 28000, episodes  28, returns 2418.03/2843.44/246.49/4494.26/5 (mean/median/min/max/num), 21.86 steps/s
2024-03-28 19:22:42,802 | TEST LOG: steps 28000, episodes  28, returns 2418.03/2843.44/246.49/4494.26/5 (mean/median/min/max/num), 21.86 steps/s
INFO:root:Normalized LOG: steps 28000, episodes  28, returns 0.65/0.75/0.14/1.15/5 (mean/median/min/max/num), 21.86 steps/s
2024-03-28 19:22:42,802 | Normalized LOG: steps 28000, episodes  28, returns 0.65/0.75/0.14/1.15/5 (mean/median/min/max/num), 21.86 steps/s
INFO:root:TRAIN LOG: steps 29000, episodes  29, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.60 steps/s
2024-03-28 19:23:29,100 | TRAIN LOG: steps 29000, episodes  29, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.60 steps/s
INFO:root:TEST LOG: steps 29000, episodes  29, returns 2243.12/1658.73/-297.78/4961.32/5 (mean/median/min/max/num), 21.60 steps/s
2024-03-28 19:23:31,424 | TEST LOG: steps 29000, episodes  29, returns 2243.12/1658.73/-297.78/4961.32/5 (mean/median/min/max/num), 21.60 steps/s
INFO:root:Normalized LOG: steps 29000, episodes  29, returns 0.61/0.47/0.01/1.26/5 (mean/median/min/max/num), 21.60 steps/s
2024-03-28 19:23:31,425 | Normalized LOG: steps 29000, episodes  29, returns 0.61/0.47/0.01/1.26/5 (mean/median/min/max/num), 21.60 steps/s
INFO:root:TRAIN LOG: steps 30000, episodes  30, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.55 steps/s
2024-03-28 19:24:17,831 | TRAIN LOG: steps 30000, episodes  30, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.55 steps/s
INFO:root:TEST LOG: steps 30000, episodes  30, returns 5031.09/5010.66/4886.35/5218.12/5 (mean/median/min/max/num), 21.55 steps/s
2024-03-28 19:24:20,340 | TEST LOG: steps 30000, episodes  30, returns 5031.09/5010.66/4886.35/5218.12/5 (mean/median/min/max/num), 21.55 steps/s
INFO:root:Normalized LOG: steps 30000, episodes  30, returns 1.27/1.27/1.24/1.32/5 (mean/median/min/max/num), 21.55 steps/s
2024-03-28 19:24:20,340 | Normalized LOG: steps 30000, episodes  30, returns 1.27/1.27/1.24/1.32/5 (mean/median/min/max/num), 21.55 steps/s
INFO:root:TRAIN LOG: steps 31000, episodes  31, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.71 steps/s
2024-03-28 19:25:06,395 | TRAIN LOG: steps 31000, episodes  31, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.71 steps/s
INFO:root:TEST LOG: steps 31000, episodes  31, returns 2846.47/2591.83/615.36/5137.57/5 (mean/median/min/max/num), 21.71 steps/s
2024-03-28 19:25:08,392 | TEST LOG: steps 31000, episodes  31, returns 2846.47/2591.83/615.36/5137.57/5 (mean/median/min/max/num), 21.71 steps/s
INFO:root:Normalized LOG: steps 31000, episodes  31, returns 0.75/0.69/0.22/1.30/5 (mean/median/min/max/num), 21.71 steps/s
2024-03-28 19:25:08,392 | Normalized LOG: steps 31000, episodes  31, returns 0.75/0.69/0.22/1.30/5 (mean/median/min/max/num), 21.71 steps/s
INFO:root:TRAIN LOG: steps 32000, episodes  32, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.88 steps/s
2024-03-28 19:25:54,089 | TRAIN LOG: steps 32000, episodes  32, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.88 steps/s
INFO:root:TEST LOG: steps 32000, episodes  32, returns 2751.28/3148.45/-10.07/5194.66/5 (mean/median/min/max/num), 21.88 steps/s
2024-03-28 19:25:56,051 | TEST LOG: steps 32000, episodes  32, returns 2751.28/3148.45/-10.07/5194.66/5 (mean/median/min/max/num), 21.88 steps/s
INFO:root:Normalized LOG: steps 32000, episodes  32, returns 0.73/0.83/0.08/1.31/5 (mean/median/min/max/num), 21.88 steps/s
2024-03-28 19:25:56,051 | Normalized LOG: steps 32000, episodes  32, returns 0.73/0.83/0.08/1.31/5 (mean/median/min/max/num), 21.88 steps/s
INFO:root:TRAIN LOG: steps 33000, episodes  33, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.96 steps/s
2024-03-28 19:26:41,596 | TRAIN LOG: steps 33000, episodes  33, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.96 steps/s
INFO:root:TEST LOG: steps 33000, episodes  33, returns 3377.61/4877.65/671.68/5292.77/5 (mean/median/min/max/num), 21.96 steps/s
2024-03-28 19:26:43,384 | TEST LOG: steps 33000, episodes  33, returns 3377.61/4877.65/671.68/5292.77/5 (mean/median/min/max/num), 21.96 steps/s
INFO:root:Normalized LOG: steps 33000, episodes  33, returns 0.88/1.24/0.24/1.34/5 (mean/median/min/max/num), 21.96 steps/s
2024-03-28 19:26:43,384 | Normalized LOG: steps 33000, episodes  33, returns 0.88/1.24/0.24/1.34/5 (mean/median/min/max/num), 21.96 steps/s
INFO:root:TRAIN LOG: steps 34000, episodes  34, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.82 steps/s
2024-03-28 19:27:29,212 | TRAIN LOG: steps 34000, episodes  34, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.82 steps/s
INFO:root:TEST LOG: steps 34000, episodes  34, returns 3052.26/4873.44/-414.33/5288.09/5 (mean/median/min/max/num), 21.82 steps/s
2024-03-28 19:27:31,304 | TEST LOG: steps 34000, episodes  34, returns 3052.26/4873.44/-414.33/5288.09/5 (mean/median/min/max/num), 21.82 steps/s
INFO:root:Normalized LOG: steps 34000, episodes  34, returns 0.80/1.24/-0.02/1.33/5 (mean/median/min/max/num), 21.82 steps/s
2024-03-28 19:27:31,304 | Normalized LOG: steps 34000, episodes  34, returns 0.80/1.24/-0.02/1.33/5 (mean/median/min/max/num), 21.82 steps/s
INFO:root:TRAIN LOG: steps 35000, episodes  35, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:28:16,349 | TRAIN LOG: steps 35000, episodes  35, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TEST LOG: steps 35000, episodes  35, returns 3210.09/3468.02/1744.04/4840.32/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:28:17,977 | TEST LOG: steps 35000, episodes  35, returns 3210.09/3468.02/1744.04/4840.32/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:Normalized LOG: steps 35000, episodes  35, returns 0.84/0.90/0.49/1.23/5 (mean/median/min/max/num), 22.20 steps/s
2024-03-28 19:28:17,977 | Normalized LOG: steps 35000, episodes  35, returns 0.84/0.90/0.49/1.23/5 (mean/median/min/max/num), 22.20 steps/s
INFO:root:TRAIN LOG: steps 36000, episodes  36, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 19:29:02,947 | TRAIN LOG: steps 36000, episodes  36, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:TEST LOG: steps 36000, episodes  36, returns 4332.03/5242.92/1689.08/5402.51/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 19:29:05,125 | TEST LOG: steps 36000, episodes  36, returns 4332.03/5242.92/1689.08/5402.51/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:Normalized LOG: steps 36000, episodes  36, returns 1.11/1.32/0.48/1.36/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 19:29:05,126 | Normalized LOG: steps 36000, episodes  36, returns 1.11/1.32/0.48/1.36/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:TRAIN LOG: steps 37000, episodes  37, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.22 steps/s
2024-03-28 19:29:50,138 | TRAIN LOG: steps 37000, episodes  37, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.22 steps/s
INFO:root:TEST LOG: steps 37000, episodes  37, returns 432.57/-532.32/-1345.70/2983.23/5 (mean/median/min/max/num), 22.22 steps/s
2024-03-28 19:29:52,278 | TEST LOG: steps 37000, episodes  37, returns 432.57/-532.32/-1345.70/2983.23/5 (mean/median/min/max/num), 22.22 steps/s
INFO:root:Normalized LOG: steps 37000, episodes  37, returns 0.18/-0.05/-0.24/0.79/5 (mean/median/min/max/num), 22.22 steps/s
2024-03-28 19:29:52,278 | Normalized LOG: steps 37000, episodes  37, returns 0.18/-0.05/-0.24/0.79/5 (mean/median/min/max/num), 22.22 steps/s
INFO:root:TRAIN LOG: steps 38000, episodes  38, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 19:30:37,233 | TRAIN LOG: steps 38000, episodes  38, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:TEST LOG: steps 38000, episodes  38, returns 2580.84/2339.30/409.49/4764.11/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 19:30:39,862 | TEST LOG: steps 38000, episodes  38, returns 2580.84/2339.30/409.49/4764.11/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:Normalized LOG: steps 38000, episodes  38, returns 0.69/0.63/0.17/1.21/5 (mean/median/min/max/num), 22.24 steps/s
2024-03-28 19:30:39,862 | Normalized LOG: steps 38000, episodes  38, returns 0.69/0.63/0.17/1.21/5 (mean/median/min/max/num), 22.24 steps/s
INFO:root:TRAIN LOG: steps 39000, episodes  39, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.21 steps/s
2024-03-28 19:31:24,883 | TRAIN LOG: steps 39000, episodes  39, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.21 steps/s
INFO:root:TEST LOG: steps 39000, episodes  39, returns 5175.95/5199.05/5092.43/5292.25/5 (mean/median/min/max/num), 22.21 steps/s
2024-03-28 19:31:27,367 | TEST LOG: steps 39000, episodes  39, returns 5175.95/5199.05/5092.43/5292.25/5 (mean/median/min/max/num), 22.21 steps/s
INFO:root:Normalized LOG: steps 39000, episodes  39, returns 1.31/1.31/1.29/1.34/5 (mean/median/min/max/num), 22.21 steps/s
2024-03-28 19:31:27,367 | Normalized LOG: steps 39000, episodes  39, returns 1.31/1.31/1.29/1.34/5 (mean/median/min/max/num), 22.21 steps/s
INFO:root:TRAIN LOG: steps 40000, episodes  40, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 19:32:12,313 | TRAIN LOG: steps 40000, episodes  40, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TEST LOG: steps 40000, episodes  40, returns 4285.86/4842.81/1921.32/5492.43/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 19:32:14,471 | TEST LOG: steps 40000, episodes  40, returns 4285.86/4842.81/1921.32/5492.43/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:Normalized LOG: steps 40000, episodes  40, returns 1.10/1.23/0.53/1.38/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 19:32:14,471 | Normalized LOG: steps 40000, episodes  40, returns 1.10/1.23/0.53/1.38/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TRAIN LOG: steps 41000, episodes  41, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.32 steps/s
2024-03-28 19:32:59,282 | TRAIN LOG: steps 41000, episodes  41, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.32 steps/s
INFO:root:TEST LOG: steps 41000, episodes  41, returns 2359.05/2063.63/77.78/4802.79/5 (mean/median/min/max/num), 22.32 steps/s
2024-03-28 19:33:01,033 | TEST LOG: steps 41000, episodes  41, returns 2359.05/2063.63/77.78/4802.79/5 (mean/median/min/max/num), 22.32 steps/s
INFO:root:Normalized LOG: steps 41000, episodes  41, returns 0.64/0.57/0.10/1.22/5 (mean/median/min/max/num), 22.32 steps/s
2024-03-28 19:33:01,033 | Normalized LOG: steps 41000, episodes  41, returns 0.64/0.57/0.10/1.22/5 (mean/median/min/max/num), 22.32 steps/s
INFO:root:TRAIN LOG: steps 42000, episodes  42, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.30 steps/s
2024-03-28 19:33:45,877 | TRAIN LOG: steps 42000, episodes  42, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.30 steps/s
INFO:root:TEST LOG: steps 42000, episodes  42, returns 3174.73/2285.19/1301.71/5351.29/5 (mean/median/min/max/num), 22.30 steps/s
2024-03-28 19:33:47,450 | TEST LOG: steps 42000, episodes  42, returns 3174.73/2285.19/1301.71/5351.29/5 (mean/median/min/max/num), 22.30 steps/s
INFO:root:Normalized LOG: steps 42000, episodes  42, returns 0.83/0.62/0.39/1.35/5 (mean/median/min/max/num), 22.30 steps/s
2024-03-28 19:33:47,450 | Normalized LOG: steps 42000, episodes  42, returns 0.83/0.62/0.39/1.35/5 (mean/median/min/max/num), 22.30 steps/s
INFO:root:TRAIN LOG: steps 43000, episodes  43, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.35 steps/s
2024-03-28 19:34:32,185 | TRAIN LOG: steps 43000, episodes  43, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.35 steps/s
INFO:root:TEST LOG: steps 43000, episodes  43, returns 3306.19/3425.34/1662.57/4794.69/5 (mean/median/min/max/num), 22.35 steps/s
2024-03-28 19:34:33,857 | TEST LOG: steps 43000, episodes  43, returns 3306.19/3425.34/1662.57/4794.69/5 (mean/median/min/max/num), 22.35 steps/s
INFO:root:Normalized LOG: steps 43000, episodes  43, returns 0.86/0.89/0.47/1.22/5 (mean/median/min/max/num), 22.35 steps/s
2024-03-28 19:34:33,858 | Normalized LOG: steps 43000, episodes  43, returns 0.86/0.89/0.47/1.22/5 (mean/median/min/max/num), 22.35 steps/s
INFO:root:TRAIN LOG: steps 44000, episodes  44, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.29 steps/s
2024-03-28 19:35:18,720 | TRAIN LOG: steps 44000, episodes  44, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.29 steps/s
INFO:root:TEST LOG: steps 44000, episodes  44, returns 3333.64/3093.89/998.11/5001.30/5 (mean/median/min/max/num), 22.29 steps/s
2024-03-28 19:35:20,580 | TEST LOG: steps 44000, episodes  44, returns 3333.64/3093.89/998.11/5001.30/5 (mean/median/min/max/num), 22.29 steps/s
INFO:root:Normalized LOG: steps 44000, episodes  44, returns 0.87/0.81/0.31/1.27/5 (mean/median/min/max/num), 22.29 steps/s
2024-03-28 19:35:20,580 | Normalized LOG: steps 44000, episodes  44, returns 0.87/0.81/0.31/1.27/5 (mean/median/min/max/num), 22.29 steps/s
INFO:root:TRAIN LOG: steps 45000, episodes  45, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 19:36:05,531 | TRAIN LOG: steps 45000, episodes  45, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TEST LOG: steps 45000, episodes  45, returns 4336.38/5077.26/1911.01/5377.62/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 19:36:07,600 | TEST LOG: steps 45000, episodes  45, returns 4336.38/5077.26/1911.01/5377.62/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:Normalized LOG: steps 45000, episodes  45, returns 1.11/1.28/0.53/1.36/5 (mean/median/min/max/num), 22.25 steps/s
2024-03-28 19:36:07,600 | Normalized LOG: steps 45000, episodes  45, returns 1.11/1.28/0.53/1.36/5 (mean/median/min/max/num), 22.25 steps/s
INFO:root:TRAIN LOG: steps 46000, episodes  46, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 19:36:52,697 | TRAIN LOG: steps 46000, episodes  46, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:TEST LOG: steps 46000, episodes  46, returns 4727.83/5193.84/2810.76/5458.52/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 19:36:55,210 | TEST LOG: steps 46000, episodes  46, returns 4727.83/5193.84/2810.76/5458.52/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:Normalized LOG: steps 46000, episodes  46, returns 1.20/1.31/0.75/1.38/5 (mean/median/min/max/num), 22.17 steps/s
2024-03-28 19:36:55,210 | Normalized LOG: steps 46000, episodes  46, returns 1.20/1.31/0.75/1.38/5 (mean/median/min/max/num), 22.17 steps/s
INFO:root:TRAIN LOG: steps 47000, episodes  47, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 19:37:40,347 | TRAIN LOG: steps 47000, episodes  47, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TEST LOG: steps 47000, episodes  47, returns 3086.42/3743.78/570.74/5252.65/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 19:37:42,323 | TEST LOG: steps 47000, episodes  47, returns 3086.42/3743.78/570.74/5252.65/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:Normalized LOG: steps 47000, episodes  47, returns 0.81/0.97/0.21/1.33/5 (mean/median/min/max/num), 22.15 steps/s
2024-03-28 19:37:42,323 | Normalized LOG: steps 47000, episodes  47, returns 0.81/0.97/0.21/1.33/5 (mean/median/min/max/num), 22.15 steps/s
INFO:root:TRAIN LOG: steps 48000, episodes  48, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 19:38:27,452 | TRAIN LOG: steps 48000, episodes  48, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TEST LOG: steps 48000, episodes  48, returns 3467.04/4305.08/1082.86/5201.14/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 19:38:29,513 | TEST LOG: steps 48000, episodes  48, returns 3467.04/4305.08/1082.86/5201.14/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:Normalized LOG: steps 48000, episodes  48, returns 0.90/1.10/0.33/1.31/5 (mean/median/min/max/num), 22.16 steps/s
2024-03-28 19:38:29,513 | Normalized LOG: steps 48000, episodes  48, returns 0.90/1.10/0.33/1.31/5 (mean/median/min/max/num), 22.16 steps/s
INFO:root:TRAIN LOG: steps 49000, episodes  49, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.04 steps/s
2024-03-28 19:39:14,881 | TRAIN LOG: steps 49000, episodes  49, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 22.04 steps/s
INFO:root:TEST LOG: steps 49000, episodes  49, returns 3914.94/4683.07/589.09/5093.83/5 (mean/median/min/max/num), 22.04 steps/s
2024-03-28 19:39:16,884 | TEST LOG: steps 49000, episodes  49, returns 3914.94/4683.07/589.09/5093.83/5 (mean/median/min/max/num), 22.04 steps/s
INFO:root:Normalized LOG: steps 49000, episodes  49, returns 1.01/1.19/0.22/1.29/5 (mean/median/min/max/num), 22.04 steps/s
2024-03-28 19:39:16,884 | Normalized LOG: steps 49000, episodes  49, returns 1.01/1.19/0.22/1.29/5 (mean/median/min/max/num), 22.04 steps/s
INFO:root:TRAIN LOG: steps 50000, episodes  50, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.78 steps/s
2024-03-28 19:40:02,801 | TRAIN LOG: steps 50000, episodes  50, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.78 steps/s
INFO:root:TEST LOG: steps 50000, episodes  50, returns 5167.27/5072.10/5001.48/5457.91/5 (mean/median/min/max/num), 21.78 steps/s
2024-03-28 19:40:05,292 | TEST LOG: steps 50000, episodes  50, returns 5167.27/5072.10/5001.48/5457.91/5 (mean/median/min/max/num), 21.78 steps/s
INFO:root:Normalized LOG: steps 50000, episodes  50, returns 1.31/1.28/1.27/1.38/5 (mean/median/min/max/num), 21.78 steps/s
2024-03-28 19:40:05,292 | Normalized LOG: steps 50000, episodes  50, returns 1.31/1.28/1.27/1.38/5 (mean/median/min/max/num), 21.78 steps/s
INFO:root:TRAIN LOG: steps 51000, episodes  51, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.65 steps/s
2024-03-28 19:40:51,478 | TRAIN LOG: steps 51000, episodes  51, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.65 steps/s
INFO:root:TEST LOG: steps 51000, episodes  51, returns 3888.89/4837.17/984.14/5157.78/5 (mean/median/min/max/num), 21.65 steps/s
2024-03-28 19:40:53,564 | TEST LOG: steps 51000, episodes  51, returns 3888.89/4837.17/984.14/5157.78/5 (mean/median/min/max/num), 21.65 steps/s
INFO:root:Normalized LOG: steps 51000, episodes  51, returns 1.00/1.23/0.31/1.30/5 (mean/median/min/max/num), 21.65 steps/s
2024-03-28 19:40:53,565 | Normalized LOG: steps 51000, episodes  51, returns 1.00/1.23/0.31/1.30/5 (mean/median/min/max/num), 21.65 steps/s
INFO:root:TRAIN LOG: steps 52000, episodes  52, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.43 steps/s
2024-03-28 19:41:40,228 | TRAIN LOG: steps 52000, episodes  52, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.43 steps/s
INFO:root:TEST LOG: steps 52000, episodes  52, returns 3997.89/5073.89/-296.97/5237.69/5 (mean/median/min/max/num), 21.43 steps/s
2024-03-28 19:41:42,932 | TEST LOG: steps 52000, episodes  52, returns 3997.89/5073.89/-296.97/5237.69/5 (mean/median/min/max/num), 21.43 steps/s
INFO:root:Normalized LOG: steps 52000, episodes  52, returns 1.03/1.28/0.01/1.32/5 (mean/median/min/max/num), 21.43 steps/s
2024-03-28 19:41:42,932 | Normalized LOG: steps 52000, episodes  52, returns 1.03/1.28/0.01/1.32/5 (mean/median/min/max/num), 21.43 steps/s
INFO:root:TRAIN LOG: steps 53000, episodes  53, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.37 steps/s
2024-03-28 19:42:29,728 | TRAIN LOG: steps 53000, episodes  53, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.37 steps/s
INFO:root:TEST LOG: steps 53000, episodes  53, returns 4439.20/5017.04/1605.67/5312.88/5 (mean/median/min/max/num), 21.37 steps/s
2024-03-28 19:42:32,036 | TEST LOG: steps 53000, episodes  53, returns 4439.20/5017.04/1605.67/5312.88/5 (mean/median/min/max/num), 21.37 steps/s
INFO:root:Normalized LOG: steps 53000, episodes  53, returns 1.13/1.27/0.46/1.34/5 (mean/median/min/max/num), 21.37 steps/s
2024-03-28 19:42:32,037 | Normalized LOG: steps 53000, episodes  53, returns 1.13/1.27/0.46/1.34/5 (mean/median/min/max/num), 21.37 steps/s
INFO:root:TRAIN LOG: steps 54000, episodes  54, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.49 steps/s
2024-03-28 19:43:18,576 | TRAIN LOG: steps 54000, episodes  54, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.49 steps/s
INFO:root:TEST LOG: steps 54000, episodes  54, returns 2067.63/547.08/-269.65/4930.37/5 (mean/median/min/max/num), 21.49 steps/s
2024-03-28 19:43:21,207 | TEST LOG: steps 54000, episodes  54, returns 2067.63/547.08/-269.65/4930.37/5 (mean/median/min/max/num), 21.49 steps/s
INFO:root:Normalized LOG: steps 54000, episodes  54, returns 0.57/0.21/0.01/1.25/5 (mean/median/min/max/num), 21.49 steps/s
2024-03-28 19:43:21,207 | Normalized LOG: steps 54000, episodes  54, returns 0.57/0.21/0.01/1.25/5 (mean/median/min/max/num), 21.49 steps/s
INFO:root:TRAIN LOG: steps 55000, episodes  55, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.95 steps/s
2024-03-28 19:44:06,772 | TRAIN LOG: steps 55000, episodes  55, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.95 steps/s
INFO:root:TEST LOG: steps 55000, episodes  55, returns 3541.83/5037.51/-1170.54/5307.60/5 (mean/median/min/max/num), 21.95 steps/s
2024-03-28 19:44:09,373 | TEST LOG: steps 55000, episodes  55, returns 3541.83/5037.51/-1170.54/5307.60/5 (mean/median/min/max/num), 21.95 steps/s
INFO:root:Normalized LOG: steps 55000, episodes  55, returns 0.92/1.28/-0.20/1.34/5 (mean/median/min/max/num), 21.95 steps/s
2024-03-28 19:44:09,373 | Normalized LOG: steps 55000, episodes  55, returns 0.92/1.28/-0.20/1.34/5 (mean/median/min/max/num), 21.95 steps/s
INFO:root:TRAIN LOG: steps 56000, episodes  56, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.72 steps/s
2024-03-28 19:44:55,405 | TRAIN LOG: steps 56000, episodes  56, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.72 steps/s
INFO:root:TEST LOG: steps 56000, episodes  56, returns 2297.59/2555.89/83.96/3956.65/5 (mean/median/min/max/num), 21.72 steps/s
2024-03-28 19:44:56,928 | TEST LOG: steps 56000, episodes  56, returns 2297.59/2555.89/83.96/3956.65/5 (mean/median/min/max/num), 21.72 steps/s
INFO:root:Normalized LOG: steps 56000, episodes  56, returns 0.62/0.69/0.10/1.02/5 (mean/median/min/max/num), 21.72 steps/s
2024-03-28 19:44:56,928 | Normalized LOG: steps 56000, episodes  56, returns 0.62/0.69/0.10/1.02/5 (mean/median/min/max/num), 21.72 steps/s
INFO:root:TRAIN LOG: steps 57000, episodes  57, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.83 steps/s
2024-03-28 19:45:42,730 | TRAIN LOG: steps 57000, episodes  57, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.83 steps/s
INFO:root:TEST LOG: steps 57000, episodes  57, returns 5158.77/5049.07/4833.05/5455.72/5 (mean/median/min/max/num), 21.83 steps/s
2024-03-28 19:45:45,321 | TEST LOG: steps 57000, episodes  57, returns 5158.77/5049.07/4833.05/5455.72/5 (mean/median/min/max/num), 21.83 steps/s
INFO:root:Normalized LOG: steps 57000, episodes  57, returns 1.30/1.28/1.23/1.37/5 (mean/median/min/max/num), 21.83 steps/s
2024-03-28 19:45:45,321 | Normalized LOG: steps 57000, episodes  57, returns 1.30/1.28/1.23/1.37/5 (mean/median/min/max/num), 21.83 steps/s
INFO:root:TRAIN LOG: steps 58000, episodes  58, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.52 steps/s
2024-03-28 19:46:31,784 | TRAIN LOG: steps 58000, episodes  58, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.52 steps/s
INFO:root:TEST LOG: steps 58000, episodes  58, returns 3636.46/4857.18/445.09/5187.57/5 (mean/median/min/max/num), 21.52 steps/s
2024-03-28 19:46:33,642 | TEST LOG: steps 58000, episodes  58, returns 3636.46/4857.18/445.09/5187.57/5 (mean/median/min/max/num), 21.52 steps/s
INFO:root:Normalized LOG: steps 58000, episodes  58, returns 0.94/1.23/0.18/1.31/5 (mean/median/min/max/num), 21.52 steps/s
2024-03-28 19:46:33,642 | Normalized LOG: steps 58000, episodes  58, returns 0.94/1.23/0.18/1.31/5 (mean/median/min/max/num), 21.52 steps/s
INFO:root:TRAIN LOG: steps 59000, episodes  59, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.52 steps/s
2024-03-28 19:47:20,118 | TRAIN LOG: steps 59000, episodes  59, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.52 steps/s
INFO:root:TEST LOG: steps 59000, episodes  59, returns 4075.83/5026.18/-71.77/5279.23/5 (mean/median/min/max/num), 21.52 steps/s
2024-03-28 19:47:22,691 | TEST LOG: steps 59000, episodes  59, returns 4075.83/5026.18/-71.77/5279.23/5 (mean/median/min/max/num), 21.52 steps/s
INFO:root:Normalized LOG: steps 59000, episodes  59, returns 1.05/1.27/0.06/1.33/5 (mean/median/min/max/num), 21.52 steps/s
2024-03-28 19:47:22,692 | Normalized LOG: steps 59000, episodes  59, returns 1.05/1.27/0.06/1.33/5 (mean/median/min/max/num), 21.52 steps/s
INFO:root:TRAIN LOG: steps 60000, episodes  60, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.49 steps/s
2024-03-28 19:48:09,228 | TRAIN LOG: steps 60000, episodes  60, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.49 steps/s
INFO:root:TEST LOG: steps 60000, episodes  60, returns 5239.57/5264.46/5093.53/5352.05/5 (mean/median/min/max/num), 21.49 steps/s
2024-03-28 19:48:12,058 | TEST LOG: steps 60000, episodes  60, returns 5239.57/5264.46/5093.53/5352.05/5 (mean/median/min/max/num), 21.49 steps/s
INFO:root:Normalized LOG: steps 60000, episodes  60, returns 1.32/1.33/1.29/1.35/5 (mean/median/min/max/num), 21.49 steps/s
2024-03-28 19:48:12,058 | Normalized LOG: steps 60000, episodes  60, returns 1.32/1.33/1.29/1.35/5 (mean/median/min/max/num), 21.49 steps/s
INFO:root:TRAIN LOG: steps 61000, episodes  61, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.47 steps/s
2024-03-28 19:48:58,639 | TRAIN LOG: steps 61000, episodes  61, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.47 steps/s
INFO:root:TEST LOG: steps 61000, episodes  61, returns 3635.45/4875.49/-594.53/5595.40/5 (mean/median/min/max/num), 21.47 steps/s
2024-03-28 19:49:01,071 | TEST LOG: steps 61000, episodes  61, returns 3635.45/4875.49/-594.53/5595.40/5 (mean/median/min/max/num), 21.47 steps/s
INFO:root:Normalized LOG: steps 61000, episodes  61, returns 0.94/1.24/-0.06/1.41/5 (mean/median/min/max/num), 21.47 steps/s
2024-03-28 19:49:01,071 | Normalized LOG: steps 61000, episodes  61, returns 0.94/1.24/-0.06/1.41/5 (mean/median/min/max/num), 21.47 steps/s
INFO:root:TRAIN LOG: steps 62000, episodes  62, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.79 steps/s
2024-03-28 19:49:46,954 | TRAIN LOG: steps 62000, episodes  62, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.79 steps/s
INFO:root:TEST LOG: steps 62000, episodes  62, returns 3533.02/4471.12/861.52/5311.37/5 (mean/median/min/max/num), 21.79 steps/s
2024-03-28 19:49:49,374 | TEST LOG: steps 62000, episodes  62, returns 3533.02/4471.12/861.52/5311.37/5 (mean/median/min/max/num), 21.79 steps/s
INFO:root:Normalized LOG: steps 62000, episodes  62, returns 0.92/1.14/0.28/1.34/5 (mean/median/min/max/num), 21.79 steps/s
2024-03-28 19:49:49,374 | Normalized LOG: steps 62000, episodes  62, returns 0.92/1.14/0.28/1.34/5 (mean/median/min/max/num), 21.79 steps/s
INFO:root:TRAIN LOG: steps 63000, episodes  63, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.71 steps/s
2024-03-28 19:50:35,443 | TRAIN LOG: steps 63000, episodes  63, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.71 steps/s
INFO:root:TEST LOG: steps 63000, episodes  63, returns 4277.05/5251.07/259.29/5463.66/5 (mean/median/min/max/num), 21.71 steps/s
2024-03-28 19:50:38,133 | TEST LOG: steps 63000, episodes  63, returns 4277.05/5251.07/259.29/5463.66/5 (mean/median/min/max/num), 21.71 steps/s
INFO:root:Normalized LOG: steps 63000, episodes  63, returns 1.09/1.33/0.14/1.38/5 (mean/median/min/max/num), 21.71 steps/s
2024-03-28 19:50:38,133 | Normalized LOG: steps 63000, episodes  63, returns 1.09/1.33/0.14/1.38/5 (mean/median/min/max/num), 21.71 steps/s
INFO:root:TRAIN LOG: steps 64000, episodes  64, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.44 steps/s
2024-03-28 19:51:24,781 | TRAIN LOG: steps 64000, episodes  64, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.44 steps/s
INFO:root:TEST LOG: steps 64000, episodes  64, returns 3592.96/5137.72/219.85/5203.10/5 (mean/median/min/max/num), 21.44 steps/s
2024-03-28 19:51:26,535 | TEST LOG: steps 64000, episodes  64, returns 3592.96/5137.72/219.85/5203.10/5 (mean/median/min/max/num), 21.44 steps/s
INFO:root:Normalized LOG: steps 64000, episodes  64, returns 0.93/1.30/0.13/1.31/5 (mean/median/min/max/num), 21.44 steps/s
2024-03-28 19:51:26,535 | Normalized LOG: steps 64000, episodes  64, returns 0.93/1.30/0.13/1.31/5 (mean/median/min/max/num), 21.44 steps/s
INFO:root:TRAIN LOG: steps 65000, episodes  65, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.51 steps/s
2024-03-28 19:52:13,033 | TRAIN LOG: steps 65000, episodes  65, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.51 steps/s
INFO:root:TEST LOG: steps 65000, episodes  65, returns 4400.53/5101.18/2477.83/5269.40/5 (mean/median/min/max/num), 21.51 steps/s
2024-03-28 19:52:15,533 | TEST LOG: steps 65000, episodes  65, returns 4400.53/5101.18/2477.83/5269.40/5 (mean/median/min/max/num), 21.51 steps/s
INFO:root:Normalized LOG: steps 65000, episodes  65, returns 1.12/1.29/0.67/1.33/5 (mean/median/min/max/num), 21.51 steps/s
2024-03-28 19:52:15,533 | Normalized LOG: steps 65000, episodes  65, returns 1.12/1.29/0.67/1.33/5 (mean/median/min/max/num), 21.51 steps/s
INFO:root:TRAIN LOG: steps 66000, episodes  66, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.44 steps/s
2024-03-28 19:53:02,176 | TRAIN LOG: steps 66000, episodes  66, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.44 steps/s
INFO:root:TEST LOG: steps 66000, episodes  66, returns 4416.60/5033.54/1570.78/5376.37/5 (mean/median/min/max/num), 21.44 steps/s
2024-03-28 19:53:04,408 | TEST LOG: steps 66000, episodes  66, returns 4416.60/5033.54/1570.78/5376.37/5 (mean/median/min/max/num), 21.44 steps/s
INFO:root:Normalized LOG: steps 66000, episodes  66, returns 1.13/1.27/0.45/1.36/5 (mean/median/min/max/num), 21.44 steps/s
2024-03-28 19:53:04,408 | Normalized LOG: steps 66000, episodes  66, returns 1.13/1.27/0.45/1.36/5 (mean/median/min/max/num), 21.44 steps/s
INFO:root:TRAIN LOG: steps 67000, episodes  67, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.38 steps/s
2024-03-28 19:53:51,175 | TRAIN LOG: steps 67000, episodes  67, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.38 steps/s
INFO:root:TEST LOG: steps 67000, episodes  67, returns 5061.63/5410.35/3703.76/5525.43/5 (mean/median/min/max/num), 21.38 steps/s
2024-03-28 19:53:53,603 | TEST LOG: steps 67000, episodes  67, returns 5061.63/5410.35/3703.76/5525.43/5 (mean/median/min/max/num), 21.38 steps/s
INFO:root:Normalized LOG: steps 67000, episodes  67, returns 1.28/1.36/0.96/1.39/5 (mean/median/min/max/num), 21.38 steps/s
2024-03-28 19:53:53,603 | Normalized LOG: steps 67000, episodes  67, returns 1.28/1.36/0.96/1.39/5 (mean/median/min/max/num), 21.38 steps/s
INFO:root:TRAIN LOG: steps 68000, episodes  68, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.46 steps/s
2024-03-28 19:54:40,207 | TRAIN LOG: steps 68000, episodes  68, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.46 steps/s
INFO:root:TEST LOG: steps 68000, episodes  68, returns 4205.06/5149.71/459.03/5213.42/5 (mean/median/min/max/num), 21.46 steps/s
2024-03-28 19:54:42,349 | TEST LOG: steps 68000, episodes  68, returns 4205.06/5149.71/459.03/5213.42/5 (mean/median/min/max/num), 21.46 steps/s
INFO:root:Normalized LOG: steps 68000, episodes  68, returns 1.08/1.30/0.19/1.32/5 (mean/median/min/max/num), 21.46 steps/s
2024-03-28 19:54:42,349 | Normalized LOG: steps 68000, episodes  68, returns 1.08/1.30/0.19/1.32/5 (mean/median/min/max/num), 21.46 steps/s
INFO:root:TRAIN LOG: steps 69000, episodes  69, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.77 steps/s
2024-03-28 19:55:30,484 | TRAIN LOG: steps 69000, episodes  69, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.77 steps/s
INFO:root:TEST LOG: steps 69000, episodes  69, returns 2001.14/800.35/-141.76/5320.05/5 (mean/median/min/max/num), 20.77 steps/s
2024-03-28 19:55:32,262 | TEST LOG: steps 69000, episodes  69, returns 2001.14/800.35/-141.76/5320.05/5 (mean/median/min/max/num), 20.77 steps/s
INFO:root:Normalized LOG: steps 69000, episodes  69, returns 0.55/0.27/0.04/1.34/5 (mean/median/min/max/num), 20.77 steps/s
2024-03-28 19:55:32,262 | Normalized LOG: steps 69000, episodes  69, returns 0.55/0.27/0.04/1.34/5 (mean/median/min/max/num), 20.77 steps/s
INFO:root:TRAIN LOG: steps 70000, episodes  70, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.32 steps/s
2024-03-28 19:56:19,156 | TRAIN LOG: steps 70000, episodes  70, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.32 steps/s
INFO:root:TEST LOG: steps 70000, episodes  70, returns 3420.81/4095.96/134.96/5309.80/5 (mean/median/min/max/num), 21.32 steps/s
2024-03-28 19:56:21,583 | TEST LOG: steps 70000, episodes  70, returns 3420.81/4095.96/134.96/5309.80/5 (mean/median/min/max/num), 21.32 steps/s
INFO:root:Normalized LOG: steps 70000, episodes  70, returns 0.89/1.05/0.11/1.34/5 (mean/median/min/max/num), 21.32 steps/s
2024-03-28 19:56:21,583 | Normalized LOG: steps 70000, episodes  70, returns 0.89/1.05/0.11/1.34/5 (mean/median/min/max/num), 21.32 steps/s
INFO:root:TRAIN LOG: steps 71000, episodes  71, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.16 steps/s
2024-03-28 19:57:08,851 | TRAIN LOG: steps 71000, episodes  71, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.16 steps/s
INFO:root:TEST LOG: steps 71000, episodes  71, returns 3668.05/3913.85/2576.15/4927.77/5 (mean/median/min/max/num), 21.16 steps/s
2024-03-28 19:57:11,080 | TEST LOG: steps 71000, episodes  71, returns 3668.05/3913.85/2576.15/4927.77/5 (mean/median/min/max/num), 21.16 steps/s
INFO:root:Normalized LOG: steps 71000, episodes  71, returns 0.95/1.01/0.69/1.25/5 (mean/median/min/max/num), 21.16 steps/s
2024-03-28 19:57:11,080 | Normalized LOG: steps 71000, episodes  71, returns 0.95/1.01/0.69/1.25/5 (mean/median/min/max/num), 21.16 steps/s
INFO:root:TRAIN LOG: steps 72000, episodes  72, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 19:57:58,588 | TRAIN LOG: steps 72000, episodes  72, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:TEST LOG: steps 72000, episodes  72, returns 3867.22/5299.79/1473.07/5518.58/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 19:58:00,879 | TEST LOG: steps 72000, episodes  72, returns 3867.22/5299.79/1473.07/5518.58/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:Normalized LOG: steps 72000, episodes  72, returns 1.00/1.34/0.43/1.39/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 19:58:00,879 | Normalized LOG: steps 72000, episodes  72, returns 1.00/1.34/0.43/1.39/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:TRAIN LOG: steps 73000, episodes  73, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.59 steps/s
2024-03-28 19:58:47,187 | TRAIN LOG: steps 73000, episodes  73, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.59 steps/s
INFO:root:TEST LOG: steps 73000, episodes  73, returns 2984.41/2050.73/793.47/5287.50/5 (mean/median/min/max/num), 21.59 steps/s
2024-03-28 19:58:49,243 | TEST LOG: steps 73000, episodes  73, returns 2984.41/2050.73/793.47/5287.50/5 (mean/median/min/max/num), 21.59 steps/s
INFO:root:Normalized LOG: steps 73000, episodes  73, returns 0.79/0.57/0.27/1.33/5 (mean/median/min/max/num), 21.59 steps/s
2024-03-28 19:58:49,243 | Normalized LOG: steps 73000, episodes  73, returns 0.79/0.57/0.27/1.33/5 (mean/median/min/max/num), 21.59 steps/s
INFO:root:TRAIN LOG: steps 74000, episodes  74, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 19:59:36,758 | TRAIN LOG: steps 74000, episodes  74, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:TEST LOG: steps 74000, episodes  74, returns 4157.61/5016.30/2544.77/5412.18/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 19:59:39,465 | TEST LOG: steps 74000, episodes  74, returns 4157.61/5016.30/2544.77/5412.18/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:Normalized LOG: steps 74000, episodes  74, returns 1.07/1.27/0.68/1.36/5 (mean/median/min/max/num), 21.05 steps/s
2024-03-28 19:59:39,465 | Normalized LOG: steps 74000, episodes  74, returns 1.07/1.27/0.68/1.36/5 (mean/median/min/max/num), 21.05 steps/s
INFO:root:TRAIN LOG: steps 75000, episodes  75, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.14 steps/s
2024-03-28 20:00:26,776 | TRAIN LOG: steps 75000, episodes  75, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.14 steps/s
INFO:root:TEST LOG: steps 75000, episodes  75, returns 4579.46/5321.44/1460.37/5495.38/5 (mean/median/min/max/num), 21.14 steps/s
2024-03-28 20:00:29,031 | TEST LOG: steps 75000, episodes  75, returns 4579.46/5321.44/1460.37/5495.38/5 (mean/median/min/max/num), 21.14 steps/s
INFO:root:Normalized LOG: steps 75000, episodes  75, returns 1.17/1.34/0.42/1.38/5 (mean/median/min/max/num), 21.14 steps/s
2024-03-28 20:00:29,032 | Normalized LOG: steps 75000, episodes  75, returns 1.17/1.34/0.42/1.38/5 (mean/median/min/max/num), 21.14 steps/s
INFO:root:TRAIN LOG: steps 76000, episodes  76, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.97 steps/s
2024-03-28 20:01:16,708 | TRAIN LOG: steps 76000, episodes  76, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.97 steps/s
INFO:root:TEST LOG: steps 76000, episodes  76, returns 2777.08/4290.79/-405.88/5010.88/5 (mean/median/min/max/num), 20.97 steps/s
2024-03-28 20:01:18,862 | TEST LOG: steps 76000, episodes  76, returns 2777.08/4290.79/-405.88/5010.88/5 (mean/median/min/max/num), 20.97 steps/s
INFO:root:Normalized LOG: steps 76000, episodes  76, returns 0.74/1.10/-0.02/1.27/5 (mean/median/min/max/num), 20.97 steps/s
2024-03-28 20:01:18,863 | Normalized LOG: steps 76000, episodes  76, returns 0.74/1.10/-0.02/1.27/5 (mean/median/min/max/num), 20.97 steps/s
INFO:root:TRAIN LOG: steps 77000, episodes  77, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.56 steps/s
2024-03-28 20:02:05,235 | TRAIN LOG: steps 77000, episodes  77, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.56 steps/s
INFO:root:TEST LOG: steps 77000, episodes  77, returns 2916.84/2804.47/-120.63/5575.05/5 (mean/median/min/max/num), 21.56 steps/s
2024-03-28 20:02:07,877 | TEST LOG: steps 77000, episodes  77, returns 2916.84/2804.47/-120.63/5575.05/5 (mean/median/min/max/num), 21.56 steps/s
INFO:root:Normalized LOG: steps 77000, episodes  77, returns 0.77/0.74/0.05/1.40/5 (mean/median/min/max/num), 21.56 steps/s
2024-03-28 20:02:07,877 | Normalized LOG: steps 77000, episodes  77, returns 0.77/0.74/0.05/1.40/5 (mean/median/min/max/num), 21.56 steps/s
INFO:root:TRAIN LOG: steps 78000, episodes  78, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.01 steps/s
2024-03-28 20:02:55,483 | TRAIN LOG: steps 78000, episodes  78, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.01 steps/s
INFO:root:TEST LOG: steps 78000, episodes  78, returns 5393.78/5339.56/5218.60/5603.58/5 (mean/median/min/max/num), 21.01 steps/s
2024-03-28 20:02:58,197 | TEST LOG: steps 78000, episodes  78, returns 5393.78/5339.56/5218.60/5603.58/5 (mean/median/min/max/num), 21.01 steps/s
INFO:root:Normalized LOG: steps 78000, episodes  78, returns 1.36/1.35/1.32/1.41/5 (mean/median/min/max/num), 21.01 steps/s
2024-03-28 20:02:58,197 | Normalized LOG: steps 78000, episodes  78, returns 1.36/1.35/1.32/1.41/5 (mean/median/min/max/num), 21.01 steps/s
INFO:root:TRAIN LOG: steps 79000, episodes  79, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.84 steps/s
2024-03-28 20:03:46,182 | TRAIN LOG: steps 79000, episodes  79, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.84 steps/s
INFO:root:TEST LOG: steps 79000, episodes  79, returns 3643.56/4928.66/1129.85/5427.51/5 (mean/median/min/max/num), 20.84 steps/s
2024-03-28 20:03:48,044 | TEST LOG: steps 79000, episodes  79, returns 3643.56/4928.66/1129.85/5427.51/5 (mean/median/min/max/num), 20.84 steps/s
INFO:root:Normalized LOG: steps 79000, episodes  79, returns 0.94/1.25/0.35/1.37/5 (mean/median/min/max/num), 20.84 steps/s
2024-03-28 20:03:48,044 | Normalized LOG: steps 79000, episodes  79, returns 0.94/1.25/0.35/1.37/5 (mean/median/min/max/num), 20.84 steps/s
INFO:root:TRAIN LOG: steps 80000, episodes  80, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.27 steps/s
2024-03-28 20:04:35,061 | TRAIN LOG: steps 80000, episodes  80, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.27 steps/s
INFO:root:TEST LOG: steps 80000, episodes  80, returns 5174.51/5169.82/5077.97/5289.23/5 (mean/median/min/max/num), 21.27 steps/s
2024-03-28 20:04:37,674 | TEST LOG: steps 80000, episodes  80, returns 5174.51/5169.82/5077.97/5289.23/5 (mean/median/min/max/num), 21.27 steps/s
INFO:root:Normalized LOG: steps 80000, episodes  80, returns 1.31/1.31/1.28/1.34/5 (mean/median/min/max/num), 21.27 steps/s
2024-03-28 20:04:37,674 | Normalized LOG: steps 80000, episodes  80, returns 1.31/1.31/1.28/1.34/5 (mean/median/min/max/num), 21.27 steps/s
INFO:root:TRAIN LOG: steps 81000, episodes  81, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.64 steps/s
2024-03-28 20:05:26,120 | TRAIN LOG: steps 81000, episodes  81, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 20.64 steps/s
INFO:root:TEST LOG: steps 81000, episodes  81, returns 3286.01/4694.22/383.15/5103.52/5 (mean/median/min/max/num), 20.64 steps/s
2024-03-28 20:05:28,268 | TEST LOG: steps 81000, episodes  81, returns 3286.01/4694.22/383.15/5103.52/5 (mean/median/min/max/num), 20.64 steps/s
INFO:root:Normalized LOG: steps 81000, episodes  81, returns 0.86/1.19/0.17/1.29/5 (mean/median/min/max/num), 20.64 steps/s
2024-03-28 20:05:28,268 | Normalized LOG: steps 81000, episodes  81, returns 0.86/1.19/0.17/1.29/5 (mean/median/min/max/num), 20.64 steps/s
INFO:root:TRAIN LOG: steps 82000, episodes  82, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.56 steps/s
2024-03-28 20:06:14,642 | TRAIN LOG: steps 82000, episodes  82, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.56 steps/s
INFO:root:TEST LOG: steps 82000, episodes  82, returns 4938.25/5241.54/3782.33/5536.45/5 (mean/median/min/max/num), 21.56 steps/s
2024-03-28 20:06:16,971 | TEST LOG: steps 82000, episodes  82, returns 4938.25/5241.54/3782.33/5536.45/5 (mean/median/min/max/num), 21.56 steps/s
INFO:root:Normalized LOG: steps 82000, episodes  82, returns 1.25/1.32/0.98/1.39/5 (mean/median/min/max/num), 21.56 steps/s
2024-03-28 20:06:16,971 | Normalized LOG: steps 82000, episodes  82, returns 1.25/1.32/0.98/1.39/5 (mean/median/min/max/num), 21.56 steps/s
INFO:root:TRAIN LOG: steps 83000, episodes  83, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.43 steps/s
2024-03-28 20:07:03,646 | TRAIN LOG: steps 83000, episodes  83, returns 0.00/0.00/0.00/0.00/5 (mean/median/min/max/num), 21.43 steps/s
INFO:root:TEST LOG: steps 83000, episodes  83, returns 3714.42/3998.01/312.48/5383.62/5 (mean/median/min/max/num), 21.43 steps/s
2024-03-28 20:07:06,400 | TEST LOG: steps 83000, episodes  83, returns 3714.42/3998.01/312.48/5383.62/5 (mean/median/min/max/num), 21.43 steps/s
INFO:root:Normalized LOG: steps 83000, episodes  83, returns 0.96/1.03/0.15/1.36/5 (mean/median/min/max/num), 21.43 steps/s
2024-03-28 20:07:06,400 | Normalized LOG: steps 83000, episodes  83, returns 0.96/1.03/0.15/1.36/5 (mean/median/min/max/num), 21.43 steps/s
"""

normalized_pattern = r"Normalized LOG: steps (\d+), episodes\s+\d+, returns (\d+\.\d+)/"
normalized_matches = re.findall(normalized_pattern, log_output)
unique_normalized_matches = list(set(normalized_matches))
normalized_data = sorted([(int(step), float(mean)) for step, mean in unique_normalized_matches], key=lambda x: x[0])

steps = [x[0] for x in normalized_data]
normalized_log = [x[1] for x in normalized_data]

plt.figure(figsize=(10, 6))
plt.plot(steps, normalized_log, marker='o', color='b')
plt.title('Normalized Log vs Iteration')
plt.xlabel('Steps (Iteration)')
plt.ylabel('Normalized Log')
plt.grid(True)
plt.show()
