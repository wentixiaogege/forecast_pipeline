#general view
推荐模式：
transport_mode: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
[1, 2, 7, 9, 11] price 存在''
[3,5,6] price全是0
4 8 10 price不为''
transport_mode
1     393982
2     246624
3     567019  price=''
4     515934  price<> ''
5     158316  price=''
6     234134  price=''
7     281512
8     15668  price<> ''
9     160854
10    103237  price<> ''
11    29345

#######
price 为'' 的比例
transport_mode
1     0.004125
2     0.004752
3     0.000000
4     0.000000 *
5     0.000000
6     0.000000
7     0.021182
8     0.000000*
9     0.000068
10    0.000000 *
11    0.002147

#1.profile_features all category
#2.od features region and cont
#3.plan features cont and words?
#4.time features morning eveting eta etc..
#5.svd feature etc...