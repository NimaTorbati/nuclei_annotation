function  masks_generator(size_target,imagej_zips_path,raw_imgs_path, results_path, num_classes)
% author: Amirreza Mahbod 
% contact: amirreza.mahbod@gmail.com

%% inputs:
% size: size of image patches (e.g. 512)
% imagej_zips: path of the created zip files from ImageJ annotation
% raw_imgs_path: raw image pathes path (.png files)
% num_classes: number of classes (e.g. 2)
%% Fuction desription: 
% creating the follwoing files form the ImageJ manual annotations
%     - raw binary masks
%     - eroded binary masks
%     - weight maps
%     - distance maps
%     - lable mask
%     - overlaid images (just for visualization)
%% file structure
% main dir --------- raw images folder (contains  'count' image patches)
%          --------- imageJ zip files  (contains 'count' zip files and each zip file contains 'n_num' roi files)
%          --------- masks                            (this directory will be created while running the code)
%          --------- mask binary                      (this directory will be created while running the code)
%          --------- mask binary without border       (this directory will be created while running the code)
%          --------- mask binary without border erode (this directory will be created while running the code)
%          --------- distance maps                    (this directory will be created while running the code)
%          --------- weighted_maps                    (this directory will be created while running the code)
%          --------- weighted_maps_erode              (this directory will be created while running the code)
%          --------- overlay                          (this directory will be created while running the code)
%          --------- nuclie border                    (this directory will be created while running the code)     




imagej_zips = dir(strcat(imagej_zips_path,'*.zip'));
raw_imgs = dir(strcat(raw_imgs_path,'*.png'));
data = {};

%% creating required dirs
for counter=1:num_classes
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','label_masks'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','mask_binary'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','mask_binary_without_border'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','mask_binary_without_border_erode'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','distance_maps'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','weighted_maps'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','weighted_maps_erode'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','overlay'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','label_masks_modify'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','stacked_mask'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk1'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk2'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk3'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk4'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk5'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk6'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk7'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk8'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk9'));
    mkdir(strcat(results_path,'class_' , string(counter),'_annotations\','nuclei_border\disk10'));
end



% main loop
object_count_tot = zeros(num_classes+1,length(imagej_zips));
for counter = 1:length(imagej_zips) %loop over images
    s = strcat(imagej_zips(counter).folder,'\',imagej_zips(counter).name);
    unzip(s, 'tempfolder');
    ROIs = dir('.\tempfolder\*.roi');
    %stacked_mask = zeros(size_target, size_target, length(ROIs));
    object_count_tot(1,counter) = length(ROIs);

    strokeColors = [];
    for n_num=1:length(ROIs) % loop over rois in one image just to determine the number of classes
           ROIName = strcat('.\tempfolder\',ROIs(n_num).name); 
           [sROI] = ReadImageJROI(ROIName);
           % Collect the nStrokeColor values
           strokeColors = [strokeColors, sROI.nStrokeColor];
    end
   % Get unique values
   uniqueStrokeColors = unique(strokeColors);
   color_opt = {'#77AC30', 'r', '#FFFFFF'}; %green = "#77AC30", '#FFFFFF' = white

   for ii = 1:length(uniqueStrokeColors)
       nuc_class_count = 0;
       mask_overlap = zeros(size_target,size_target);
       mask_overlap_modify = zeros(size_target,size_target);
       mask_overlap_borderremove = zeros(size_target,size_target);
       D_overlap = zeros(size_target, size_target);
       for n_num=1:length(ROIs) % loop over rois in one image
           ROIName = strcat('.\tempfolder\',ROIs(n_num).name); 
           [sROI] = ReadImageJROI(ROIName);
           if sROI.nStrokeColor == uniqueStrokeColors(ii)
               nuc_class_count = nuc_class_count+1;
               mask = poly2mask(sROI.mnCoordinates(:,1),sROI.mnCoordinates(:,2), 512, 512);
               %stacked_mask(:,:,n_num)= mask;
               mask_org = mask;
               D = bwdist(~mask);
               mask = double(mask)*n_num;
               mask_overlap_modify = max(mask, mask_overlap_modify);
               mask_overlap = mask + mask_overlap;
               D_overlap = max(D ,D_overlap);
               %% for border remove
               eshterak = find(mask_overlap_borderremove == mask_org & mask_overlap_borderremove==1);
               eshterak_img = zeros(512,512);
               eshterak_img(eshterak)=1; 
               mask_overlap_borderremove = max(mask_org, mask_overlap_borderremove);
               if length(eshterak)~=0
                   B = edge(mask_org,'nothinning');
                   thin_edge = edge(mask_org);
                   se = strel('disk', 1);
                   B = B & eshterak_img;
                   B2 = imdilate(B,strel(se));
           
                   mask_overlap_borderremove = mask_overlap_borderremove - B2; 
                   mask_overlap_borderremove = mask_overlap_borderremove - thin_edge; 
                   mask_overlap_borderremove(mask_overlap_borderremove==-1)=0;
               end
           end
       end
       object_count_tot (ii + 1, counter) = nuc_class_count;
               %% for masks with diffent label for each object    
               savepath_labelmask = strcat(results_path,'class_' , string(ii),'_annotations\','label_masks','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               imwrite(uint16(mask_overlap),savepath_labelmask);
               %% for masks with diffent label for each object (overlaying areas assign to one object!)    
               savepath_labelmask = strcat(results_path,'class_' , string(ii),'_annotations\','label_masks_modify','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               imwrite(uint16(mask_overlap_modify),savepath_labelmask);
               %% for binary mask
               mask_binary = zeros(512,512);
               mask_binary (mask_overlap>0)= 255;
               mask_binary = uint8(mask_binary);
               savepath_binary = strcat(results_path,'class_' , string(ii),'_annotations\','mask_binary','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               imwrite(mask_binary,savepath_binary);

               %% for nuclie border
               for diskthick =1:10
                   se = strel('disk', diskthick);
                   eroded_mask_binary = imerode(mask_binary,strel(se));
                   border = mask_binary - eroded_mask_binary; 
                   savepath_border = strcat(results_path,'class_' , string(ii),'_annotations\','nuclei_border\disk',string(diskthick),'\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
                   imwrite(border, savepath_border);
               end

               %% for mask removing borders like TMI paper
               savepath_binary_borderremoved = strcat(results_path,'class_' , string(ii),'_annotations\','mask_binary_without_border','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               mask_overlap_borderremove (mask_overlap_borderremove>0)= 255;
               imwrite(uint8(mask_overlap_borderremove),savepath_binary_borderremoved);
    
               %% for weighted maps
               gt = mask_overlap_borderremove; 
               se = strel('disk', 1);
               gt_erode = imerode(gt,strel(se));
    
               [weight]=unetwmap(gt);
               [weight_erode]=unetwmap(gt_erode);
               weight = weight* 255/max(weight(:));
               weight_erode = weight_erode* 255/max(weight_erode(:));
    
               weighted_maps_path = strcat(results_path,'class_' , string(ii),'_annotations\','weighted_maps','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               weighted_maps_erode_path = strcat(results_path,'class_' , string(ii),'_annotations\','weighted_maps_erode','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               imwrite(uint8(weight),weighted_maps_path,'Mode','lossless');
               imwrite(uint8(weight_erode),weighted_maps_erode_path,'Mode','lossless');
    
               %% to save the math files (not needed)
               % savepath_math = strcat(results_path,'math_weighted_maps','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               % savepath_math_erode = strcat(results_path,'math_weighted_maps_erode','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               % save(savepath_math,'weight')
               % save(savepath_math_erode,'weight_erode')
    
               %% for mask binary without border erode
               se = strel('disk', 1);
               mask_binary_without_border_erode = imerode(mask_overlap_borderremove,strel(se));
               savepath_binary_borderremoved = strcat(results_path,'class_' , string(ii),'_annotations\','mask_binary_without_border_erode','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               imwrite(uint8(mask_binary_without_border_erode),savepath_binary_borderremoved);
    
               %% for distance maps
               savepath_distance = strcat(results_path,'class_' , string(ii),'_annotations\','distance_maps','\', strcat(erase(raw_imgs(counter).name,'.png'),'.png'));
               D_overlap = double(D_overlap);
               D_overlap = D_overlap/max(D_overlap(:)); % otherwise you get bianry image for distance map
               imwrite(D_overlap,savepath_distance,'Mode','lossless');
    
               %% for overliad images
               original= imread(strcat(raw_imgs(counter).folder,'\',raw_imgs(counter).name));
               img_dim = size(original);
               if length(img_dim) ==2
                   original = cat(3, original, original, original);
               end
               original_r = original(:,:,1);
               original_g = original(:,:,2);
               original_b = original(:,:,3);

               original_r(mask_overlap~=0) = 255;
               original_g(mask_overlap~=0) = 255;
               original_b(mask_overlap~=0) = 255;

               original2(:,:,1) = original_r;
               original2(:,:,2) = original_g;
               original2(:,:,3) = original_b;
               fig = figure('Renderer', 'painters', 'Position', [10 10 1500 750]);
               %subplot(1,2,2);imshow(original2);%title({'overlay'},'FontSize', 22); %with white fill in
               subplot(1,2,2);imshow(original); title({'overlay'},'FontSize', 22);
               for i=1:length(ROIs)
                   dum = mask_overlap;
                   dum(mask_overlap~=i)=0;
                   hold on 
                   visboundaries(dum,'Color',color_opt{ii},'LineWidth', 1,'LineStyle', '-');
               end
               subplot(1,2,1);imshow(original);title({'cropped image'},'FontSize', 22);
    
               save_path_overlay = strcat(results_path,'class_' , string(ii),'_annotations\','overlay','\',raw_imgs(counter).name);
               % for white border (otherwise the borders will turn balcj after saving)
               if color_opt{ii}=='#FFFFFF'
                   set(gcf, 'InvertHardCopy', 'off');
               end
               saveas(fig, save_path_overlay);
    
               mask_overlap = zeros(size_target,size_target);
               mask_overlap_modify = zeros(size_target,size_target);
               mask_overlap_borderremove = zeros(size_target,size_target);
               D_overlap = zeros(512,512);

    close all
   end
   fprintf('%s has totally %d nuclei\n', raw_imgs(counter).name, object_count_tot(1, counter));
   fprintf('==========\n')
   delete tempfolder\*.roi
   data = [data; {raw_imgs(counter).name, object_count_tot(1, counter), object_count_tot(2, counter), object_count_tot(3, counter), object_count_tot(4, counter)}];
   end

   fprintf('All images have %d nuclei\n', sum(object_count_tot(1,:)));
   for i=2:size(object_count_tot,1)
       fprintf('All images have %d nuclei of class %d \n', sum(object_count_tot(i,:)), i-1);
   end

   % Calculate the sum of each column (excluding the name)
   sum_value1 = sum(cell2mat(data(:,2)));
   sum_value2 = sum(cell2mat(data(:,3)));
   sum_value3 = sum(cell2mat(data(:,4)));

   % Append the sum as the last row (with a label 'Total')
   data = [data; {'Total', sum_value1, sum_value2, sum_value3, sum_value3}];

   % Convert the cell array to a table
   T = cell2table(data, 'VariableNames', {'Name', 'total_nuclei', 'class1_nuclei', 'class2_nuclei', 'class3_nuclei'});

   % Write the table to a CSV file
   writetable(T, strcat(results_path ,'stats.csv'));

   % Display the table (optional)
   disp(T);
   
   end

