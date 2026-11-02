# **Paper 6: Privacy-Utility Trade-off in Social Location Analytics under India's DPDP Act**

Module 6: Social Media Location, Search Engine Analytics and Applications

Type: Method and evaluation

## **Research Gaps**

* Privacy mechanisms are usually compared in isolation, on different tasks and datasets.

* Legal obligations under the DPDP Act are rarely operationalised into technical evaluation criteria.

* Many studies report utility only and omit attack-based privacy evaluation.

* Systematic comparison of federated learning with and without differential privacy for POI tasks is limited.

## **Key Contributions**

* A unified comparison of geo-indistinguishability, cloaking, DP-SGD and federated learning on shared tasks and datasets.

* Privacy-utility Pareto curves paired with re-identification and membership-inference results.

* A DPDP compliance checklist mapped to technical controls.

* Design guidelines and open code for compliant location analytics.

## **Draft Abstract**

Location analytics supports mobility, urban and public-safety applications, but privacy techniques are usually compared on technical metrics without linkage to legal obligations such as India's Digital Personal Data Protection (DPDP) Act. This paper compares geo-indistinguishability, spatial cloaking, DP-SGD and federated learning (with and without differential privacy) on next-POI prediction and hotspot detection using Gowalla, Brightkite, Foursquare, Geolife and Porto taxi data. Privacy budgets are swept and utility is reported alongside re-identification and membership-inference attack success. Results are mapped to a DPDP compliance checklist covering consent, purpose limitation and data minimisation. Findings show \[utility retained at moderate epsilon\] and that \[mechanism\] offers the best trade-off, yielding design guidelines for compliant location analytics.

*Bracketed values are placeholders to be filled after experiments are complete.*

## **Section-by-Section Outline**

### **1\. Introduction**

* Location data value and risk; DPDP Act context

* Gap: no unified comparison of mechanisms mapped to legal obligations

* Research questions and contributions

### **2\. Related Work**

* Location privacy: k-anonymity, cloaking, geo-indistinguishability

* Differential privacy and federated learning for mobility models

* Privacy attacks on trajectories; regulation-aware system design

### **3\. Problem Formulation**

* Threat model; utility tasks; privacy budget definition

### **4\. Framework**

* Mechanisms and configurations

* Attack suite (re-identification, membership inference)

* DPDP compliance checklist and mapping method (legal expert review)

### **5\. Experimental Setup**

* Datasets: Gowalla, Brightkite, Foursquare NYC/Tokyo, Geolife, Porto taxi, T-Drive

* Baselines: raw data, k-anonymity only, non-private centralised model

* Metrics: Acc@k, hotspot F1, epsilon, attack success rate, communication cost

### **6\. Results**

* Privacy-utility Pareto curves

* Attack success versus budget

* Federated versus centralised comparison

* Ablation and sensitivity

### **7\. Discussion**

* Recommended mechanism by use case; compliance implications

### **8\. Limitations**

* Legal interpretation is indicative; dataset geography; simulated federation

### **9\. Conclusion and Future Work**

* Real-time social sensing; fairness across user groups

## **Expected Figures and Diagrams**

*Keep captions short; place technical detail in axis labels, legends and in-figure labels.*

| No. | Short caption | Type and content |
| :---- | :---- | :---- |
| Fig. 1 | Framework overview | Pipeline: data, mechanisms, tasks, attacks, compliance mapping |
| Fig. 2 | Threat model | Diagram of adversary capabilities and data flows |
| Fig. 3 | Mechanism placement | Where each mechanism acts (client, server, model) |
| Fig. 4 | Federated setup | Client-server training diagram with DP noise |
| Fig. 5 | Privacy-utility curves | Acc@k and hotspot F1 versus epsilon per mechanism |
| Fig. 6 | Attack success | Bars or lines of attack success versus budget |
| Fig. 7 | Pareto front | Utility versus privacy risk across mechanisms |
| Fig. 8 | Spatial noise example | Map view of raw versus perturbed locations |
| Fig. 9 | DPDP mapping | Matrix linking obligations to technical controls |

## **Related-Work Keyword List**

| Theme | Search strings |
| :---- | :---- |
| Location privacy | location privacy; geo-indistinguishability; spatial cloaking; k-anonymity trajectory |
| DP and FL | differential privacy trajectory data; DP-SGD; federated learning next POI; federated mobility prediction |
| Attacks | membership inference attack; trajectory re-identification; location inference attack |
| Tasks | next POI recommendation; hotspot detection; human mobility prediction |
| Regulation | DPDP Act 2023; GDPR location data; privacy by design compliance; privacy-utility trade-off |

**References (30) last 5 years**  
