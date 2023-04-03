

# required libraries
from numpy import allclose
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
# from nltk.stem import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
# import math
from sklearn.preprocessing import normalize
from optht import optht
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import scipy.sparse as scsp


#########

class Data():


    def __init__(self, document, query, proof):
        self.document = document
        self.query = query
        self.proof = proof

    

    #### #### #### DOCUMENTS
    # def read_doc(self, document):
    def read_doc(self):
        # self.log_file = open(document)
        # with self.log_file as f:
        with open(self.document) as f:

            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
    
        # Put each DOCUMENT into a dictionary
        doc_set = {}
        doc_id = ""
        doc_text = ""
        for l in lines:
            if l.startswith(".I"):
                doc_id = l.split(" ")[1].strip()
            elif l.startswith(".X"):
                doc_set[doc_id] = doc_text.lstrip(" ")
                doc_id = ""
                doc_text = ""
            else:
                doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.
        return doc_set



    #### #### #### QUERIES
    def read_query(self):

        with open(self.query) as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
            
        qry_set = {}
        qry_id = ""
        for l in lines:
            if l.startswith(".I"):
                qry_id = l.split(" ")[1].strip()
            elif l.startswith(".W"):
                qry_set[qry_id] = l.strip()[3:]
                qry_id = ""  
        # print(f"Number of queries = {len(qry_set)}" + ".\n")
        return qry_set



    #### #### #### Ground Proofs
    def read_ground_proof(self):

        rel_set = {}
        with open(self.proof) as f:
            for l in f.readlines():
                qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
                doc_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]
                if qry_id in rel_set:
                    rel_set[qry_id].append(doc_id)
                else:
                    rel_set[qry_id] = []
                    rel_set[qry_id].append(doc_id)
                # if qry_id == "7":
                #     print(l.strip("\n"))
            
        # Print something to see the dictionary structure, etc.
        # print(f"\nNumber of mappings = {len(rel_set)}" + ".\n")
        # print('Documents related to query 11:')
        # print(rel_set["1"]) # note that the dictionary indexes are strings, not numbers.
            
        return   rel_set

    # Merged list of documents and queries. Renaming Ground Proofs keys' name to be consistent with queries' names. 
    def call_doc_query_ground_proof(self):
        qrys = self.read_query()
        docs = self.read_doc()
        gproofs = self.read_ground_proof()

        # Drop queries which are not included in Ground Proof set
        for q in list(qrys.keys()):
            if q not in list(gproofs.keys()):
             del qrys[q]
        docs_qrys = list(docs.values()) + list(qrys.values())

        # Renaming Ground Proofs keys' name to be consistent with queries' names
        keys = np.linspace(1,len(qrys), len(qrys), dtype= int)
        values = list(gproofs.values())
        gp = {}
        for key in keys:
            gp[str(key)] = values[key-1]

        return docs_qrys, gp  # Add Queries list to the Documents list




class Preprocess():




    def __init__(self, docs_qrys):
        self.docs_qrys = docs_qrys
    
    # Create DataFrame of all Documents and Queries
    def df_doc_query(self):

        vect = self.CountVectorizer()  
        df= self.pd.DataFrame(vect.fit_transform(self.docs_qrys).toarray().transpose(), index = vect.get_feature_names())
        df.drop(df.index[:381], inplace=True) # Drop non-alphabetic
        return df

    #  Stop-Words and Common-Words
    def stop_words(self):
        stop_words = set(self.stop_words.words("english"))
        commen_words = set(open('commen_words.txt', 'r').read().split('\n'))
        stop_common_list = [ ind for ind in self.df_doc_query().index if ind in stop_words | commen_words]
        return self.df_doc_query().drop(stop_common_list) # Remove stop words and common words


    # Term weighting scheme
    def weighted(self):
        tdm = self.stem()
        list_n_i = self.np.count_nonzero(tdm, axis=1) # list all n_i's
        return self.pd.DataFrame([tdm.iloc[term,:]* self.math.log10(tdm.shape[1]/list_n_i[term]) for term in range(len(tdm.index))])
    # stemming
    def stem(self):
        df = self.stop_words()
        stemmer = self.PorterStemmer()
        stemmed_indices = [stemmer.stem(ind) for ind in df.index ] # Set of stemmed indices
        df.index= stemmed_indices # Replace indices labels with stemmed indices
        return df.groupby(df.index).sum()



# /// Class of Informatino Retrivial's algorithems 
class InformationRetrievalAlgo():
    # import pandas as pd
    def __init__(self, tdm, gp): # term-document matrix and modified ground proofs
        self.tdm = tdm
        self.docs = tdm.loc[:,:1459]
        self.gp = gp # modified ground proofs


    # Full vector matrix
    def full_vecvotr_matrix(self,query):
        qry = self.tdm[1459+query]
        # tols = tolerance
        gproof =  [ int(x)-1 for x in self.gp[str(query)]]
        # gproof = map(lambda x: int(x), self.rel_set[str(query)]) # [ int(x) -1 for x in self.rel_set[str(query)]]

        # Matching Performance
        match_ = []
        for doc in self.docs:
            q_d_j = self.np.dot(qry.T, self.docs.loc[:,doc])
            q_norm2 = self.np.linalg.norm(qry, ord=2)
            d_j_norm2= np.linalg.norm(self.docs.iloc[:,doc], ord = 2)

            cos = q_d_j / (q_norm2 * d_j_norm2) # Cosine distance
            match_.append((doc,cos))

        df_match = pd.DataFrame.from_records(match_, columns=['doc', 'cos'])  # convert list of touples into dictionary

        _prec = []
        precision = np.array(_prec, dtype = np.float32)
        _rec = []
        recall = np.array(_rec, dtype = np.float32)
        tols = np.arange(0.05, df_match['cos'].max(), .001)
        for tol in tols:

            d_t = df_match[df_match['cos'] >= tol].reset_index(drop=True)               # Total numebr of documents retrieved
            n_r = gproof                                                            # Total number of relevant documents in the database
            d_r = [ d_t.loc[x,'doc'] for x in range(len(d_t)) if d_t.loc[x,'doc'] in n_r] # Number of relevant documents 


            precision =np.append(precision, len(d_r)/ len(d_t) ) # Precision
            recall = np.append(recall, len(d_r)/ len(n_r)) # Recall

        return recall, precision


    ### Latent Semantic Indexing (SVD)
    def lsi_model(self,query, rank = 'opt'): # Rank can be chosen, otherwise by default rank is equal to "opt".


        # Calling Query and related Ground Proof
        qry = self.tdm[1459+query]
        gproof =  [ int(x)-1 for x in self.gp[str(query)]]

        # Normalize and SVD
        tdm_norm = normalize(self.tdm, norm= 'l2',axis=0 )
        q = normalize(np.array(qry).reshape(1,-1), norm='l2', axis=0).ravel()

        [u, s, v] = np.linalg.svd(tdm_norm, full_matrices= False)


        if rank == 'opt':
            opt = optht(tdm_norm, sv= s, sigma= None)

            # Reduced u, s and v
            U_k = u[:, range(opt)]
            V_k = v[range(opt), :]
            S_k = np.diag(s[range(opt)])

            # H_k and tdm approximation
            H_k = np.dot(S_k, V_k)
            tdm_k = np.dot(U_k, H_k)
            
            # Error
            apprx_error = []
            apprx_error.append(np.linalg.norm(tdm_norm - tdm_k, 'fro') / np.linalg.norm(tdm_norm, 'fro')) # Martix approximation erroe
            
            # q_k
            q_k = np.dot(U_k.T,q )
        else:
            # Reduced u, s and v
            U_k = u[:, range(rank)]
            V_k = v[range(rank), :]
            S_k = np.diag(s[range(rank)])

            # H_k and tdm approximation
            H_k = np.dot(S_k, V_k)
            tdm_k = np.dot(U_k, H_k)
            
            # Error
            apprx_error = []
            apprx_error.append(np.linalg.norm(tdm_norm - tdm_k, 'fro') / np.linalg.norm(tdm_norm, 'fro')) # Martix approximation erroe
            
            # q_k
            q_k = np.dot(U_k.T,q )

        # Cos similarity
        j = 0
        match = []
        for j in range( H_k.shape[1]-1):
            q_k_h_j = np.dot(q_k.T, H_k[:,j])
            q_k_norm2 = np.linalg.norm(q_k, ord=2)
            h_j_norm2= np.linalg.norm(H_k[:,j], ord = 2)


            cos = q_k_h_j / (q_k_norm2 * h_j_norm2)
            match.append((j,cos))

        df_match = pd.DataFrame.from_records(match, columns=['doc', 'cos'])  # convert list of touples into dictionary
        
        # Recall and Precision
        _prec = []
        precision = np.array(_prec, dtype = np.float32)
        _rec = []
        recall = np.array(_rec, dtype = np.float32)
        tols = np.linspace(0.05, df_match['cos'].max(), 100)
        for tol in tols:
            d_t = df_match[df_match['cos'] >= tol].reset_index(drop=True)
            
            n_r = gproof
            d_r = [d_t.loc[x,'doc'] for x in range(len(d_t)) if d_t.loc[x,'doc'] in gproof]
            precision = np.append(precision, len(d_r)/ len(d_t))
            recall = np.append(recall, len(d_r)/ len(n_r))
        
        
        return recall, precision, apprx_error


    ### Clustering
    def clustering_model(self, query, num_cluster):



        # Calling Query and related Ground Proof
        qry = self.tdm[1459+query]
        gproof =  [ int(x)-1 for x in self.gp[str(query)]]

        # Normalize
        tdm_norm = normalize(self.tdm, norm= 'l2',axis=0 )
        tdm_norm = pd.DataFrame(tdm_norm, columns= self.tdm.columns)

        q = normalize(np.array(qry).reshape(1,-1), norm='l2', axis=0).ravel()

        # Centroids
        tdm_kmeans = KMeans(n_clusters= num_cluster).fit(tdm_norm.T)
        
        # QR Decompostion of centroids
        C = tdm_kmeans.cluster_centers_.T
        P, R = np.linalg.qr(C)
        G = np.dot(P.T, tdm_norm.iloc[:,:1460] )
        q_k = np.dot(P.T,q )

        # Error
        apprx_error = []
        apprx_error = np.linalg.norm(tdm_norm.loc[:,:1459] - np.dot(P, G), 'fro') / np.linalg.norm(tdm_norm.loc[:,:1459], 'fro') # Martix approximation erroe
        
        # Cos similarity
        match_ = []
        for doc in range(G.shape[1]-1):
            q_d_j = np.dot(q_k.T, G[:,doc])
            q_norm2 = np.linalg.norm(q_k, ord=2)
            d_j_norm2= np.linalg.norm(G[:,doc], ord = 2)


            cos = q_d_j / (q_norm2 * d_j_norm2) # Cosine distance
            match_.append((doc,cos))

        df_match = pd.DataFrame.from_records(match_, columns=['doc', 'cos'])  # convert list of touples into dictionary

        _prec = []
        precision = np.array(_prec, dtype = np.float32)
        _rec = []
        recall = np.array(_rec, dtype = np.float32)
        tols = np.linspace(0.05, df_match['cos'].max(), 100)

        for tol in tols:

            d_t = df_match[df_match['cos'] >= tol].reset_index(drop=True)               # Total numebr of documents retrieved
            n_r =  gproof                                                             # Total number of relevant documents in the database
            d_r = [ d_t.loc[x,'doc'] for x in range(len(d_t)) if d_t.loc[x,'doc'] in n_r] # Number of relevant documents 


            precision =np.append(precision, len(d_r)/ len(d_t) ) # Precision
            recall = np.append(recall, len(d_r)/ len(n_r)) # Recall

        
        
        return recall, precision, apprx_error



    ### Nonnegative Matrix Factorization
    def nmf_model(self,query,num_components ): # num_components actually means number of topics

        # Calling Query and related Ground Proof
        qry = self.tdm[1459+query]
        gproof =  [ int(x)-1 for x in self.gp[str(query)]]

        # Fitt model and approximate W and H
        model = NMF(n_components= num_components, init='random', random_state=0)
        W = model.fit_transform(self.tdm.loc[:,:1459])
        H = model.components_

        # OR decomposition of W and inverse of R
        Q, R = np.linalg.qr(W)
        R_inv = np.linalg.inv(R)

        # Query in the reduced basis
        q_hat = np.dot(R_inv, np.dot(Q.T, qry))

        # Cos similarity
        j = 0
        match = []
        for j in range( H.shape[1]-1):
            #  print(j)
            q_hat_h_j = np.dot(q_hat.T, H[:,j])
            q_hat_norm2 = np.linalg.norm(q_hat, ord=2)
            h_j_norm2= np.linalg.norm(H[:,j], ord = 2)


            cos = q_hat_h_j / (q_hat_norm2 * h_j_norm2)
            match.append((j,cos))

        df_match = pd.DataFrame.from_records(match, columns=['doc', 'cos'])  # convert list of touples into dictionary

        _prec_k_nmf = []
        precision = np.array(_prec_k_nmf, dtype = np.float32)
        _rec_k_nmf = []
        recall = np.array(_rec_k_nmf, dtype = np.float32)

        tols = np.linspace(0.05, df_match['cos'].max(), 100)
        for tol in tols:
            d_t = df_match[df_match['cos'] >= tol].reset_index(drop=True)
            n_r = gproof
            d_r = [d_t.loc[x,'doc'] for x in range(len(d_t)) if d_t.loc[x,'doc'] in n_r]
            precision = np.append(precision, len(d_r)/ len(d_t))
            recall = np.append(recall, len(d_r)/ len(n_r))
        prec_rec_nmf = list(zip(recall, precision, tols))
        df_nmf = pd.DataFrame(prec_rec_nmf,columns=[ 'recall', 'precision', 'tol'] )

        apprx_error = []
        apprx_error = np.linalg.norm(self.tdm.loc[:,:1459] - np.dot(W, H), 'fro') / np.linalg.norm(self.tdm.loc[:,:1459], 'fro') # Martix approximation erroe
        
        return recall, precision, apprx_error

    
    
    ### LGK Bidiagonalization

    def lgkb_model(self, query, step):


        # Calling Query and related Ground Proof
        qry = self.tdm[1459+query]
        gproof =  [ int(x)-1 for x in self.gp[str(query)]]

        A = np.matrix(self.tdm.loc[:,:1459])
        m,n = A.shape
        
        alpha = np.zeros(step)
        beta = np.zeros(step+1)
        P = np.zeros((step+1, m))  # in fact it's P.H
        Z = np.zeros((step, n))  # in fact it's Z.H
        # Transposed matrices are used for easier slicing
        P[0] = (qry / np.linalg.norm(qry, ord=2))
        
        for i in range(step):
            Z[i] = A.H@P[i] - beta[i]*Z[i-1]
            alpha[i] = np.linalg.norm(Z[i])
            Z[i] /= alpha[i]
            P[i+1] = A@Z[i] - alpha[i]*P[i]
            beta[i+1] = np.linalg.norm(P[i+1])
            P[i+1] /= beta[i+1]

        P,Z = map(np.matrix, (P, Z))
        P = P.H
        B = (alpha, beta[1:])
        Z = Z.H  
        B = scsp.diags(B, [0, -1] ,shape = (step+1,step)).toarray()
        Q, B_hat_zero = np.linalg.qr(B, mode= 'complete') 
        I = np.vstack((np.identity(B_hat_zero.shape[1]), np.zeros(B_hat_zero.shape[1])))
        B_hat = B_hat_zero[0:step]
        W_k = P@Q@I
        Y_k = Z@B_hat.T
        q_tilde = W_k@W_k.T@qry


    ##Cosine Similarity
        
        match_ = []
        G = self.tdm.loc[:,:1459]
        for doc in range(self.tdm.loc[:,:1459].shape[1]-1):
            q_d_j = np.dot(q_tilde.T, G.loc[:,doc])
            q_norm2 = np.linalg.norm(q_tilde, ord=2)
            d_j_norm2= np.linalg.norm(G.loc[:,doc], ord = 2)


            cos = q_d_j / (q_norm2 * d_j_norm2) # Cosine distance
            match_.append((doc,cos))

        df_match = pd.DataFrame.from_records(match_, columns=['doc', 'cos'])  # convert list of touples into dictionary

        _prec = []
        precision = np.array(_prec, dtype = np.float32)
        _rec = []
        recall = np.array(_rec, dtype = np.float32)

        tols = np.linspace(0.05, df_match['cos'].max(), 100)
        for tol in tols:

            d_t = df_match[df_match['cos'] >= tol].reset_index(drop=True)               # Total numebr of documents retrieved
            n_r =  gproof                                                             # Total number of relevant documents in the database
            d_r = [ d_t.loc[x,'doc'] for x in range(len(d_t)) if d_t.loc[x,'doc'] in n_r] # Number of relevant documents 


            precision =np.append(precision, len(d_r)/ len(d_t) ) # Precision
            recall = np.append(recall, len(d_r)/ len(n_r)) # Recall

        # Error
        apprx_error = []
        apprx_error= np.linalg.norm(self.tdm.loc[:,:1459] - P@B@Z.T, 'fro') / np.linalg.norm(self.tdm.loc[:,:1459], 'fro') # Martix approximation erroe
    
        
        return recall, precision, apprx_error




class AveragePerformance(InformationRetrievalAlgo):

    def average_full_vector_matrix(self, number_of_queries):
        all_qrys = []
        for q in range(number_of_queries):
            [rec,prec] =  InformationRetrievalAlgo.full_vecvotr_matrix(self,q+1)
            all_qrys.append([rec,prec])
        return self.for_display(number_of_queries,all_qrys)



    def average_lsi_model(self,number_of_queries,rank):
        all_qrys = []
        for q in range(number_of_queries):
            rec,prec,apprx_error =  InformationRetrievalAlgo.lsi_model(self,q+1,rank)
            all_qrys.append([rec,prec])
        return self.for_display(number_of_queries,all_qrys)



    def average_clustering_model(self,number_of_queries,num_cluster):
        all_qrys = []
        for q in range(number_of_queries):
            rec,prec,apprx_error =  InformationRetrievalAlgo.clustering_model(self,q+1,num_cluster)
            all_qrys.append([rec,prec])
        return self.for_display(number_of_queries,all_qrys)



    def average_nmf_model(self,number_of_queries,num_components):
        all_qrys = []
        for q in range(number_of_queries):
            rec,prec,apprx_error =  InformationRetrievalAlgo.nmf_model(self,q+1,num_components)
            all_qrys.append([rec,prec])
        
        return self.for_display(number_of_queries,all_qrys)



    def average_lgkb_model(self,number_of_queries,step):
        all_qrys = []
        for q in range(number_of_queries):
            rec,prec,apprx_error =  InformationRetrievalAlgo.lgkb_model(self,q+1,step)
            all_qrys.append([rec,prec])
        return self.for_display(number_of_queries,all_qrys)



    def for_display(self, number_of_queries,all_queries):
        points = np.arange(.05,.95,.05)
        rec_intpol =[]
        prec_intpol = []
        for i in range(number_of_queries):

            interpolate = interp1d(all_queries[i][0],all_queries[i][1])

            start_point = next(points[j] for j in range(len(points)) if min(all_queries[i][0]) < points[j] )
            stop_point = next(points[len(points)-1-j] for j in range(len(points)) if max(all_queries[i][0]) > points[len(points)-j-1])
            
            rec_intpol.append([round(num,2) for num in list(np.arange(start_point,stop_point+.001, .05))])
            prec_intpol.append([round(num,3) for num in interpolate(rec_intpol[i]).tolist()])
        

            gap_down = round(((min(rec_intpol[0]) - .05)*100) /5)
            rec_intpol[i] = list(np.hstack((np.zeros(gap_down), rec_intpol[i] )))
            prec_intpol[i] = list(np.hstack((np.zeros(gap_down) + np.nan, prec_intpol[i] )))


            gap_up = round(((.9 - max(rec_intpol[i]))*100) /5)
            rec_intpol[i] = list(np.hstack((rec_intpol[i],np.zeros(gap_up))))
            prec_intpol[i] = list(np.hstack((prec_intpol[i],np.zeros(gap_up) + np.nan )))

        recall_points = points
        precision_values = pd.DataFrame(prec_intpol).mean()
        return recall_points, precision_values