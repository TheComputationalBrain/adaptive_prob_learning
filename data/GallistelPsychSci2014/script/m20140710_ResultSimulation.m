% Plot the results of the simulation of Gallistel model vs. ideal observer:
% is there an effect of the pre-jump duration on the data?

fname = 'simGallistel_7_10_2_54';
load(fname)

mean_Ga = squeeze(nanmean(FirstDetect_Ga, 1));
mean_FB = squeeze(nanmean(FirstDetect_FB, 1));

sem_Ga = squeeze(stderror(FirstDetect_Ga,1));
sem_FB = squeeze(stderror(FirstDetect_FB,1));

figure(1); clf; set(gcf, 'Color', [1 1 1])

errorbar(L1-LpreTest, mean_FB(1,:), sem_FB(1,:), 'b')
hold on
errorbar(L1-LpreTest, mean_FB(2,:), sem_FB(2,:), 'b--')
errorbar(L1-LpreTest, mean_Ga(1,:), sem_Ga(1,:), 'g')
errorbar(L1-LpreTest, mean_Ga(2,:), sem_Ga(2,:), 'g--')

legend({...
    sprintf('Ideal Observer p=%3.2f -> %3.2f', p(1), 1-p(1)), ...
    sprintf('Ideal Observer p=%3.2f -> %3.2f', p(2), 1-p(2)), ...
    sprintf('Change point   p=%3.2f -> %3.2f', p(1), 1-p(1)), ...
    sprintf('Change point   p=%3.2f -> %3.2f', p(2), 1-p(2)) ...
    }, 'Location', 'NorthEastOutside')

errorbar(L1-LpreTest, mean_FB(1,:), sem_FB(1,:), '.b')
errorbar(L1-LpreTest, mean_FB(2,:), sem_FB(2,:), '.b')
errorbar(L1-LpreTest, mean_Ga(1,:), sem_Ga(1,:), '.g')
errorbar(L1-LpreTest, mean_Ga(2,:), sem_Ga(2,:), '.g')

xlabel('Duration of pre-jump period (in samples)')
ylabel({'Latency of 1st detection'; ...
    sprintf('(mean over %d simulations)', nSim)})
title(fname, 'Interpreter', 'none')

hgexport(1, '~/ResultGalistelSimulations.eps')