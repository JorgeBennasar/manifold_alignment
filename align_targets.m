function [targets_1_aligned,targets_2_aligned] = ...
    align_targets(targets_1,index_1,targets_2,index_2)

targets_1_aligned = targets_1;

vec_2_aligned = zeros(1,8);
for i = 1:8
    vec_2_aligned(i) = find(index_2 == index_1(i));
end

targets_2_aligned = zeros(1,length(targets_2));
for i = 1:length(targets_2)
    targets_2_aligned(i) = find(vec_2_aligned == targets_2(i));
end

end