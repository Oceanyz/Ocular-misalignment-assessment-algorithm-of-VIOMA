# Ocular-misalignment-assessment-algorithm-of-VIOMA
A video analysis-based ocular misalignment assessment algorithm 

The strabismus video from our self-built StrabismusACT-76, as the input to the algorithm, is processed to output the strabismus assessment results, including the presence or absence of strabismus, the type, and the magnitude of strabismus. In the algorithm module, the eye regions are firstly extracted, then iris boundaries are detected, with diameters measured. The purpose of key frame detection is to find the position where the cover event of interest occurs, by detecting pre-set marker. The pupil is localized in each frame within an interval of the key frame. The estimation of ocular misalignment is based on the variation in pupil center positions at the starting and ending positions.

If you want to use the assessment algorithm before the article is published, please contact by hfu@eduhk.hk or zhengy@xidian.edu.cn, and cite the following articles:

Zheng Y, Fu H, Li B, Lo W-L, Wen D. An automatic stimulus and synchronous tracking system for strabismus assessment based on cover test. In: Proc 2018 International Conference on Intelligent Informatics and Biomedical Sciences (ICIIBMS), 2018. pp 123-127. 
Zheng Y, Fu H, Li R, Lo W-L, Chi Z, Feng DD, Song Z, Wen D. Intelligent evaluation of strabismus in videos based on an automated cover test. Applied Sciences 2019;9(4):731.
