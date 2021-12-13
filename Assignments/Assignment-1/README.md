## Brief Overview of the Approach:
- Bayes A: Assumed the same covariance matrix (Identity Matrix) and calculated the ML Estimate of the mean along each dimension. Used this ML estimate to make predictions on new test points.
- Bayes B: Assumed the same covariance matrix (A Matrix Sigma) and calculated the ML Estimate of the mean and Covariance along each dimension. Used these ML estimates to make predictions on new test points.
- Bayes C: Assumed the same covariance matrix for each class (Positive and Negative for this Binary Classifier) and calculated the ML Estimate of the mean, Covariance Matrix for each class along each dimension. Used these ML estimates to make predictions on new test points.
 
 
## Running the Scripts:
- Ensure that the dataset added to Q5 for testing is in the Q5 Folder before running the script with the given command

'''
cd Q5
python3 CE19B110_NA19B030.py
'''

- Note that all the plots are generated at once and this script takes approximately 30 seconds to generate results with a system having 16GB RAM and an Intel i7 10th Gen Processor.


