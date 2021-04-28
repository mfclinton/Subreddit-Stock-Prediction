#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob

path = r'../../data/' # use your path
all_submission_files = glob.glob(path + "*_submission.csv")

sb_li = []

for filename in all_submission_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    sb_li.append(df)

submissions = pd.concat(sb_li, axis=0, ignore_index=True)


# In[2]:


all_comments_files = glob.glob(path + "*_comments.csv")

co_li = []

for filename in all_comments_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    co_li.append(df)

comments = pd.concat(co_li, axis=0, ignore_index=True)




# Includes US stock symbols with market cap > 100 Million, and price above $3. 
# Download the csv file  https://www.nasdaq.com/market-activity/stocks/screener?exchange=nasdaq&letter=0&render=download 
# of all the NYSE, NASDAQ and NYSEAMERICAN public traded companies.

# In[5]:


US_STOCK_SYMBOLS = {
    'A', 'AA', 'AACQ', 'AAIC', 'AAL', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAT', 'AAWW', 'AAXN', 'AB', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABCL', 'ABCM', 'ABEV', 'ABG', 'ABM', 'ABMD', 'ABNB', 'ABR', 'ABST', 'ABT', 'ABTX', 'ABUS', 'AC', 'ACA', 'ACAC', 'ACAD', 'ACAM', 'ACB', 'ACBI', 'ACC', 'ACCD', 'ACCO', 'ACEL', 'ACET', 'ACEV', 'ACGL', 'ACH', 'ACHC', 'ACI', 'ACIA', 'ACIC', 'ACIU', 'ACIW', 'ACLS', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACND', 'ACP', 'ACRE', 'ACRS', 'ACTC', 'ACTG', 'ACU', 'ACV', 'ADAP', 'ADBE', 'ADC', 'ADCT', 'ADES', 'ADI', 'ADM', 'ADMS', 'ADNT', 'ADOC', 'ADP', 'ADPT', 'ADS', 'ADSK', 'ADT', 'ADTN', 'ADUS', 'ADV', 'ADVM', 'ADX', 'AE', 'AEE', 'AEF', 'AEG', 'AEGN', 'AEIS', 'AEL', 'AEM', 'AEO', 'AEP', 'AER', 'AERI', 'AES', 'AEYE', 'AFB', 'AFG', 'AFIB', 'AFIN', 'AFL', 'AFMD', 'AFT', 'AFYA', 'AG', 'AGC', 'AGCO', 'AGD', 'AGEN', 'AGI', 'AGIO', 'AGLE', 'AGM', 'AGMH', 'AGNC', 'AGO', 'AGR', 'AGRO', 'AGS', 'AGTC', 'AGX', 'AGYS', 'AHAC', 'AHCO', 'AHH', 'AI', 'AIF', 'AIG', 'AIH', 'AIMC', 'AIN', 'AINV', 'AIO', 'AIR', 'AIRC', 'AIRG', 'AIT', 'AIV', 'AIZ', 'AJAX', 'AJG', 'AJRD', 'AJX', 'AKAM', 'AKR', 'AKRO', 'AKTS', 'AKU', 'AKUS', 'AL', 'ALB', 'ALBO', 'ALC', 'ALCO', 'ALDX', 'ALE', 'ALEC', 'ALEX', 'ALG', 'ALGM', 'ALGN', 'ALGS', 'ALGT', 'ALK', 'ALKS', 'ALL', 'ALLE', 'ALLK', 'ALLO', 'ALLT', 'ALLY', 'ALNY', 'ALPN', 'ALRM', 'ALRS', 'ALSK', 'ALSN', 'ALT', 'ALTA', 'ALTG', 'ALTM', 'ALTR', 'ALUS', 'ALV', 'ALVR', 'ALX', 'ALXN', 'ALXO', 'AM', 'AMAL', 'AMAT', 'AMBA', 'AMBC', 'AMCI', 'AMCR', 'AMCX', 'AMD', 'AME', 'AMED', 'AMEH', 'AMG', 'AMGN', 'AMH', 'AMHC', 'AMK', 'AMKR', 'AMN', 'AMNB', 'AMOT', 'AMOV', 'AMP', 'AMPH', 'AMRC', 'AMRK', 'AMRN', 'AMRS', 'AMRX', 'AMSC', 'AMSF', 'AMSWA', 'AMT', 'AMTB', 'AMTBB', 'AMTI', 'AMWD', 'AMWL', 'AMX', 'AMYT', 'AMZN', 'AN', 'ANAB', 'ANAT', 'ANDE', 'ANET', 'ANF', 'ANGI', 'ANGO', 'ANIK', 'ANIP', 'ANNX', 'ANSS', 'ANTM', 'AOD', 'AON', 'AONE', 'AOS', 'AOSL', 'AOUT', 'AP', 'APA', 'APAM', 'APD', 'APEI', 'APG', 'APH', 'APHA', 'API', 'APLE', 'APLS', 'APLT', 'APO', 'APOG', 'APPF', 'APPN', 'APPS', 'APRE', 'APSG', 'APT', 'APTO', 'APTS', 'APTV', 'APTX', 'APVO', 'APXT', 'APYX', 'AQB', 'AQMS', 'AQN', 'AQST', 'AQUA', 'AR', 'ARA', 'ARAY', 'ARCB', 'ARCC', 'ARCE', 'ARCH', 'ARCO', 'ARCT', 'ARD', 'ARDC', 'ARDX', 'ARE', 'ARES', 'ARGO', 'ARGX', 'ARI', 'ARL', 'ARLO', 'ARLP', 'ARMK', 'ARNA', 'ARNC', 'AROC', 'AROW', 'ARQT', 'ARR', 'ARRY', 'ARTNA', 'ARVN', 'ARW', 'ARWR', 'ARYA', 'ASA', 'ASAN', 'ASAQ', 'ASB', 'ASC', 'ASG', 'ASGN', 'ASH', 'ASIX', 'ASLE', 'ASMB', 'ASML', 'ASND', 'ASO', 'ASPL', 'ASPN', 'ASPS', 'ASPU', 'ASR', 'ASTE', 'ASUR', 'ASX', 'ATAC', 'ATAX', 'ATCO', 'ATEC', 'ATEN', 'ATEX', 'ATGE', 'ATH', 'ATHA', 'ATHM', 'ATI', 'ATKR', 'ATLC', 'ATLO', 'ATNI', 'ATNM', 'ATNX', 'ATO', 'ATOM', 'ATR', 'ATRA', 'ATRC', 'ATRI', 'ATRO', 'ATRS', 'ATSG', 'ATTO', 'ATUS', 'ATVI', 'AU', 'AUB', 'AUBN', 'AUDC', 'AUPH', 'AUTL', 'AUY', 'AVA', 'AVAL', 'AVAN', 'AVAV', 'AVB', 'AVCT', 'AVD', 'AVDL', 'AVEO', 'AVGO', 'AVID', 'AVIR', 'AVK', 'AVLR', 'AVNS', 'AVNT', 'AVNW', 'AVO', 'AVRO', 'AVT', 'AVTR', 'AVXL', 'AVY', 'AVYA', 'AWF', 'AWH', 'AWI', 'AWK', 'AWP', 'AWR', 'AX', 'AXDX', 'AXGN', 'AXL', 'AXLA', 'AXNX', 'AXP', 'AXS', 'AXSM', 'AXTA', 'AXTI', 'AXU', 'AY', 'AYI', 'AYLA', 'AYRO', 'AYTU', 'AYX', 'AZEK', 'AZN', 'AZO', 'AZPN', 'AZRE', 'AZUL', 'AZYO', 'AZZ', 'B', 'BA', 'BABA', 'BAC', 'BAF', 'BAH', 'BAK', 'BALY', 'BAM', 'BANC', 'BAND', 'BANF', 'BANR', 'BANX', 'BAP', 'BASI', 'BATL', 'BATRA', 'BATRK', 'BAX', 'BB', 'BBAR', 'BBBY', 'BBCP', 'BBD', 'BBDC', 'BBDO', 'BBF', 'BBIO', 'BBK', 'BBL', 'BBN', 'BBSI', 'BBU', 'BBVA', 'BBY', 'BC', 'BCAB', 'BCAT', 'BCBP', 'BCC', 'BCE', 'BCEI', 'BCEL', 'BCH', 'BCLI', 'BCML', 'BCO', 'BCOR', 'BCOV', 'BCPC', 'BCRX', 'BCS', 'BCSF', 'BCTG', 'BCV', 'BCX', 'BCYC', 'BDC', 'BDGE', 'BDJ', 'BDN', 'BDSI', 'BDSX', 'BDTX', 'BDX', 'BE', 'BEAM', 'BEAT', 'BECN', 'BEDU', 'BEEM', 'BEKE', 'BELFA', 'BELFB', 'BEN', 'BEP', 'BEPC', 'BERY', 'BFAM', 'BFC', 'BFI', 'BFIN', 'BFK', 'BFRA', 'BFS', 'BFST', 'BFT', 'BFZ', 'BG', 'BGB', 'BGCP', 'BGFV', 'BGH', 'BGIO', 'BGNE', 'BGR', 'BGS', 'BGSF', 'BGT', 'BGX', 'BGY', 'BH', 'BHB', 'BHC', 'BHE', 'BHF', 'BHK', 'BHLB', 'BHP', 'BHR', 'BHVN', 'BIDU', 'BIF', 'BIG', 'BIGC', 'BIIB', 'BILI', 'BILL', 'BIO', 'BIOX', 'BIP', 'BIPC', 'BIT', 'BIVI', 'BJ', 'BJRI', 'BK', 'BKD', 'BKE', 'BKH', 'BKI', 'BKN', 'BKNG', 'BKR', 'BKT', 'BKU', 'BL', 'BLBD', 'BLCT', 'BLD', 'BLDP', 'BLDR', 'BLE', 'BLFS', 'BLI', 'BLK', 'BLKB', 'BLL', 'BLMN', 'BLNK', 'BLSA', 'BLU', 'BLUE', 'BLW', 'BLX', 'BMA', 'BMCH', 'BME', 'BMI', 'BMO', 'BMRC', 'BMRN', 'BMTC', 'BMY', 'BNED', 'BNFT', 'BNGO', 'BNL', 'BNR', 'BNS', 'BNTX', 'BNY', 'BOAC', 'BOCH', 'BOE', 'BOH', 'BOKF', 'BOMN', 'BOOM', 'BOOT', 'BOWX', 'BOX', 'BP', 'BPFH', 'BPMC', 'BPMP', 'BPOP', 'BPRN', 'BPY', 'BPYU', 'BQ', 'BR', 'BRBR', 'BRBS', 'BRC', 'BREZ', 'BRFS', 'BRG', 'BRID', 'BRKL', 'BRKR', 'BRKS', 'BRMK', 'BRO', 'BROG', 'BRP', 'BRT', 'BRX', 'BRY', 'BSAC', 'BSBK', 'BSBR', 'BSD', 'BSET', 'BSGM', 'BSIG', 'BSL', 'BSM', 'BSMX', 'BSRR', 'BST', 'BSTZ', 'BSVN', 'BSX', 'BSY', 'BTA', 'BTAI', 'BTAQ', 'BTAQU', 'BTBT', 'BTG', 'BTI', 'BTO', 'BTT', 'BTWN', 'BTZ', 'BUD', 'BUFF', 'BUI', 'BUR', 'BURL', 'BUSE', 'BV', 'BVH', 'BVN', 'BW', 'BWA', 'BWB', 'BWEN', 'BWFG', 'BWG', 'BWMX', 'BWXT', 'BX', 'BXC', 'BXG', 'BXMT', 'BXMX', 'BXP', 'BXS', 'BY', 'BYD', 'BYM', 'BYND', 'BYSI', 'BZH', 'BZUN', 'C', 'CAAP', 'CAAS', 'CABA', 'CABO', 'CAC', 'CACC', 'CACI', 'CADE', 'CAE', 'CAF', 'CAG', 'CAH', 'CAI', 'CAJ', 'CAKE', 'CAL', 'CALA', 'CALB', 'CALM', 'CALT', 'CALX', 'CAMP', 'CAMT', 'CAN', 'CANG', 'CAPA', 'CAPL', 'CAR', 'CARA', 'CARE', 'CARG', 'CARR', 'CARS', 'CASA', 'CASH', 'CASS', 'CASY', 'CAT', 'CATC', 'CATM', 'CATO', 'CATY', 'CB', 'CBAN', 'CBAT', 'CBAY', 'CBB', 'CBD', 'CBFV', 'CBH', 'CBIO', 'CBMG', 'CBNK', 'CBOE', 'CBPO', 'CBRE', 'CBRL', 'CBSH', 'CBT', 'CBTX', 'CBU', 'CBZ', 'CC', 'CCAC', 'CCAP', 'CCB', 'CCBG', 'CCC', 'CCCC', 'CCD', 'CCEP', 'CCF', 'CCI', 'CCIV', 'CCJ', 'CCK', 'CCL', 'CCMP', 'CCNE', 'CCOI', 'CCRN', 'CCS', 'CCU', 'CCX', 'CCXI', 'CD', 'CDAK', 'CDAY', 'CDE', 'CDK', 'CDLX', 'CDMO', 'CDNA', 'CDNS', 'CDR', 'CDW', 'CDXC', 'CDXS', 'CDZI', 'CE', 'CEA', 'CECE', 'CEE', 'CEIX', 'CEL', 'CELH', 'CEM', 'CENT', 'CENTA', 'CENX', 'CEO', 'CEQP', 'CERE', 'CERN', 'CERS', 'CERT', 'CET', 'CEVA', 'CF', 'CFB', 'CFBK', 'CFFI', 'CFFN', 'CFG', 'CFII', 'CFR', 'CFRX', 'CFX', 'CG', 'CGBD', 'CGC', 'CGEN', 'CGNX', 'CGO', 'CGRO', 'CHA', 'CHAQ', 'CHCO', 'CHCT', 'CHD', 'CHDN', 'CHE', 'CHEF', 'CHGG', 'CHH', 'CHI', 'CHKP', 'CHL', 'CHMA', 'CHMG', 'CHMI', 'CHN', 'CHNG', 'CHPM', 'CHRS', 'CHRW', 'CHT', 'CHTR', 'CHU', 'CHUY', 'CHW', 'CHWY', 'CHX', 'CHY', 'CI', 'CIA', 'CIB', 'CIEN', 'CIGI', 'CII', 'CIIC', 'CIK', 'CIM', 'CINF', 'CINR', 'CIO', 'CIR', 'CIT', 'CIVB', 'CIX', 'CIXX', 'CIZN', 'CKH', 'CL', 'CLA', 'CLAR', 'CLB', 'CLBK', 'CLCT', 'CLDR', 'CLDT', 'CLDX', 'CLF', 'CLFD', 'CLGX', 'CLH', 'CLI', 'CLII', 'CLLS', 'CLM', 'CLMT', 'CLNC', 'CLNE', 'CLNY', 'CLPR', 'CLPT', 'CLR', 'CLS', 'CLSK', 'CLVR', 'CLVS', 'CLW', 'CLX', 'CLXT', 'CM', 'CMA', 'CMBM', 'CMC', 'CMCL', 'CMCO', 'CMCSA', 'CMCT', 'CMD', 'CME', 'CMG', 'CMI', 'CMLF', 'CMLS', 'CMO', 'CMP', 'CMPI', 'CMPR', 'CMPS', 'CMRE', 'CMRX', 'CMS', 'CMT', 'CMTL', 'CMU', 'CNA', 'CNBKA', 'CNC', 'CNCE', 'CNDT', 'CNF', 'CNHI', 'CNI', 'CNK', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNP', 'CNQ', 'CNR', 'CNS', 'CNSL', 'CNST', 'CNTG', 'CNTY', 'CNX', 'CNXC', 'CNXN', 'CO', 'CODI', 'CODX', 'COE', 'COF', 'COFS', 'COG', 'COGT', 'COHR', 'COHU', 'COKE', 'COLB', 'COLD', 'COLL', 'COLM', 'COMM', 'CONE', 'CONN', 'COO', 'COOP', 'COP', 'COR', 'CORE', 'CORT', 'COST', 'COTY', 'COUP', 'COWN', 'CP', 'CPA', 'CPAC', 'CPB', 'CPE', 'CPF', 'CPK', 'CPLG', 'CPLP', 'CPRI', 'CPRT', 'CPRX', 'CPS', 'CPSI', 'CPSR', 'CPST', 'CPT', 'CQP', 'CR', 'CRAI', 'CRC', 'CRDF', 'CREE', 'CRESY', 'CRF', 'CRH', 'CRHC', 'CRI', 'CRIS', 'CRK', 'CRL', 'CRM', 'CRMD', 'CRMT', 'CRNC', 'CRNX', 'CRON', 'CROX', 'CRS', 'CRSA', 'CRSP', 'CRSR', 'CRTO', 'CRTX', 'CRUS', 'CRVL', 'CRWD', 'CRY', 'CS', 'CSBR', 'CSCO', 'CSGP', 'CSGS', 'CSII', 'CSIQ', 'CSL', 'CSOD', 'CSPR', 'CSQ', 'CSR', 'CSSE', 'CSTE', 'CSTL', 'CSTM', 'CSTR', 'CSV', 'CSWC', 'CSWI', 'CSX', 'CTAC', 'CTAS', 'CTB', 'CTBI', 'CTIC', 'CTLT', 'CTMX', 'CTO', 'CTR', 'CTRA', 'CTRE', 'CTRN', 'CTS', 'CTSH', 'CTSO', 'CTT', 'CTVA', 'CTXS', 'CUB', 'CUBE', 'CUBI', 'CUE', 'CUK', 'CULP', 'CURI', 'CURO', 'CUTR', 'CUZ', 'CVA', 'CVAC', 'CVBF', 'CVCO', 'CVCY', 'CVE', 'CVEO', 'CVET', 'CVGI', 'CVGW', 'CVI', 'CVLB', 'CVLG', 'CVLT', 'CVLY', 'CVM', 'CVNA', 'CVS', 'CVX', 'CW', 'CWCO', 'CWEN', 'CWH', 'CWK', 'CWST', 'CWT', 'CX', 'CXDO', 'CXE', 'CXO', 'CXP', 'CXW', 'CYAD', 'CYBE', 'CYBR', 'CYCN', 'CYD', 'CYH', 'CYRX', 'CYTK', 'CZNC', 'CZR', 'CZWI', 'CZZ', 'D', 'DAC', 'DADA', 'DAKT', 'DAL', 'DAN', 'DAO', 'DAR', 'DASH', 'DAVA', 'DB', 'DBD', 'DBDR', 'DBI', 'DBL', 'DBX', 'DCBO', 'DCF', 'DCI', 'DCO', 'DCOM', 'DCP', 'DCPH', 'DCRB', 'DCT', 'DCTH', 'DD', 'DDD', 'DDOG', 'DDS', 'DE', 'DEA', 'DECK', 'DEH', 'DEI', 'DELL', 'DEN', 'DENN', 'DEO', 'DESP', 'DEX', 'DFHT', 'DFIN', 'DFNS', 'DFP', 'DFPH', 'DFS', 'DG', 'DGICA', 'DGICB', 'DGII', 'DGNR', 'DGNS', 'DGX', 'DHC', 'DHF', 'DHI', 'DHIL', 'DHR', 'DHT', 'DIAX', 'DIN', 'DIOD', 'DIS', 'DISCA', 'DISCB', 'DISCK', 'DISH', 'DJCO', 'DK', 'DKL', 'DKNG', 'DKS', 'DL', 'DLA', 'DLB', 'DLHC', 'DLR', 'DLTH', 'DLTR', 'DLX', 'DM', 'DMAC', 'DMB', 'DMF', 'DMLP', 'DMO', 'DMRC', 'DMS', 'DMTK', 'DMYD', 'DNB', 'DNLI', 'DNOW', 'DNP', 'DOC', 'DOCU', 'DOMO', 'DOOO', 'DOOR', 'DORM', 'DOV', 'DOW', 'DOX', 'DOYU', 'DPG', 'DPZ', 'DQ', 'DRD', 'DRE', 'DRH', 'DRI', 'DRIO', 'DRNA', 'DRQ', 'DSAC', 'DSGX', 'DSKE', 'DSL', 'DSM', 'DSPG', 'DSSI', 'DSU', 'DT', 'DTE', 'DTF', 'DTIL', 'DUC', 'DUK', 'DUO', 'DVA', 'DVAX', 'DVN', 'DX', 'DXC', 'DXCM', 'DXPE', 'DY', 'DYAI', 'DYN', 'DZSI', 'E', 'EA', 'EAD', 'EAF', 'EAR', 'EARN', 'EAT', 'EB', 'EBAY', 'EBC', 'EBF', 'EBIX', 'EBMT', 'EBON', 'EBR', 'EBS', 'EBSB', 'EBTC', 'EC', 'ECC', 'ECF', 'ECHO', 'ECL', 'ECOL', 'ECOM', 'ECPG', 'ED', 'EDAP', 'EDD', 'EDF', 'EDIT', 'EDN', 'EDU', 'EDUC', 'EEFT', 'EEX', 'EFC', 'EFF', 'EFL', 'EFR', 'EFSC', 'EFT', 'EFX', 'EGAN', 'EGBN', 'EGHT', 'EGLE', 'EGO', 'EGOV', 'EGP', 'EGRX', 'EH', 'EHC', 'EHI', 'EHT', 'EHTH', 'EIDX', 'EIG', 'EIGI', 'EIGR', 'EIM', 'EIX', 'EL', 'ELA', 'ELAN', 'ELF', 'ELLO', 'ELOX', 'ELP', 'ELS', 'ELVT', 'ELY', 'EMD', 'EME', 'EMF', 'EMKR', 'EML', 'EMN', 'EMO', 'EMPW', 'EMR', 'EMX', 'ENB', 'ENBL', 'ENDP', 'ENIA', 'ENIC', 'ENLC', 'ENLV', 'ENPC', 'ENPH', 'ENR', 'ENS', 'ENSG', 'ENTA', 'ENTG', 'ENV', 'ENVA', 'ENX', 'EOD', 'EOG', 'EOI', 'EOLS', 'EOS', 'EOSE', 'EOT', 'EPAC', 'EPAM', 'EPAY', 'EPC', 'EPD', 'EPIX', 'EPR', 'EPRT', 'EPZM', 'EQ', 'EQBK', 'EQC', 'EQD', 'EQH', 'EQIX', 'EQNR', 'EQOS', 'EQR', 'EQT', 'EQX', 'ERC', 'ERES', 'ERF', 'ERH', 'ERIC', 'ERIE', 'ERII', 'ERJ', 'ERYP', 'ES', 'ESCA', 'ESE', 'ESGR', 'ESI', 'ESLT', 'ESNT', 'ESPR', 'ESQ', 'ESRT', 'ESS', 'ESSA', 'ESSC', 'ESTA', 'ESTC', 'ESTE', 'ESXB', 'ET', 'ETAC', 'ETB', 'ETG', 'ETH', 'ETJ', 'ETN', 'ETNB', 'ETO', 'ETON', 'ETR', 'ETRN', 'ETSY', 'ETV', 'ETW', 'ETX', 'ETY', 'EURN', 'EV', 'EVA', 'EVBG', 'EVBN', 'EVER', 'EVF', 'EVG', 'EVGN', 'EVH', 'EVI', 'EVLO', 'EVM', 'EVN', 'EVOP', 'EVR', 'EVRG', 'EVRI', 'EVT', 'EVTC', 'EVV', 'EW', 'EWBC', 'EXAS', 'EXC', 'EXEL', 'EXFO', 'EXG', 'EXK', 'EXLS', 'EXP', 'EXPC', 'EXPD', 'EXPE', 'EXPI', 'EXPO', 'EXR', 'EXTN', 'EXTR', 'EYE', 'EYEN', 'EZPW', 'F', 'FAF', 'FAII', 'FAM', 'FANG', 'FANH', 'FARO', 'FAST', 'FATE', 'FAX', 'FB', 'FBC', 'FBHS', 'FBIO', 'FBIZ', 'FBK', 'FBM', 'FBMS', 'FBNC', 'FBP', 'FBRX', 'FC', 'FCAC', 'FCAP', 'FCAU', 'FCBC', 'FCBP', 'FCCO', 'FCCY', 'FCEL', 'FCF', 'FCFS', 'FCN', 'FCNCA', 'FCPT', 'FCRD', 'FCT', 'FCX', 'FDBC', 'FDEU', 'FDMT', 'FDP', 'FDS', 'FDUS', 'FDX', 'FE', 'FEI', 'FEIM', 'FELE', 'FEN', 'FENC', 'FEYE', 'FF', 'FFA', 'FFBC', 'FFC', 'FFG', 'FFIC', 'FFIN', 'FFIV', 'FFNW', 'FFWM', 'FGBI', 'FGEN', 'FHB', 'FHI', 'FHN', 'FHTX', 'FIBK', 'FICO', 'FIF', 'FIII', 'FINS', 'FIS', 'FISI', 'FISV', 'FIT', 'FITB', 'FIV', 'FIVE', 'FIVN', 'FIX', 'FIXX', 'FIZZ', 'FL', 'FLC', 'FLDM', 'FLEX', 'FLGT', 'FLIC', 'FLIR', 'FLL', 'FLMN', 'FLNG', 'FLNT', 'FLO', 'FLOW', 'FLR', 'FLS', 'FLT', 'FLUX', 'FLWS', 'FLXN', 'FLXS', 'FLY', 'FMAC', 'FMAO', 'FMBH', 'FMBI', 'FMC', 'FMN', 'FMNB', 'FMS', 'FMTX', 'FMX', 'FN', 'FNB', 'FNCB', 'FND', 'FNF', 'FNKO', 'FNLC', 'FNV', 'FNWB', 'FOCS', 'FOE', 'FOF', 'FOLD', 'FONR', 'FOR', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FOUR', 'FOX', 'FOXA', 'FOXF', 'FPF', 'FPH', 'FPI', 'FPL', 'FPRX', 'FR', 'FRA', 'FRAF', 'FRBA', 'FRC', 'FREE', 'FREQ', 'FRG', 'FRGI', 'FRHC', 'FRLN', 'FRME', 'FRO', 'FROG', 'FRPH', 'FRPT', 'FRSX', 'FRT', 'FRTA', 'FSBW', 'FSD', 'FSDC', 'FSFG', 'FSK', 'FSKR', 'FSLF', 'FSLR', 'FSLY', 'FSM', 'FSP', 'FSR', 'FSRV', 'FSS', 'FST', 'FSTR', 'FSV', 'FT', 'FTAI', 'FTCH', 'FTDR', 'FTF', 'FTHM', 'FTI', 'FTIV', 'FTNT', 'FTOC', 'FTS', 'FTSI', 'FTV', 'FUBO', 'FUL', 'FULC', 'FULT', 'FUN', 'FUNC', 'FUND', 'FUSE', 'FUSN', 'FUTU', 'FUV', 'FVAM', 'FVCB', 'FVE', 'FVRR', 'FWONA', 'FWONK', 'FWRD', 'G', 'GAB', 'GABC', 'GAIA', 'GAIN', 'GAM', 'GAN', 'GATO', 'GATX', 'GB', 'GBAB', 'GBCI', 'GBDC', 'GBIO', 'GBL', 'GBLI', 'GBT', 'GBX', 'GCBC', 'GCI', 'GCMG', 'GCO', 'GCP', 'GCV', 'GD', 'GDDY', 'GDEN', 'GDL', 'GDO', 'GDOT', 'GDP', 'GDRX', 'GDS', 'GDV', 'GDYN', 'GE', 'GEF', 'GEL', 'GENC', 'GEO', 'GEOS', 'GES', 'GEVO', 'GF', 'GFF', 'GFI', 'GFL', 'GFN', 'GGAL', 'GGB', 'GGG', 'GGM', 'GGN', 'GGT', 'GGZ', 'GH', 'GHC', 'GHG', 'GHIV', 'GHL', 'GHLD', 'GHM', 'GHY', 'GIB', 'GIII', 'GIK', 'GIL', 'GILD', 'GILT', 'GIM', 'GIS', 'GIX', 'GKOS', 'GL', 'GLAD', 'GLDD', 'GLEO', 'GLNG', 'GLO', 'GLOB', 'GLOG', 'GLP', 'GLPG', 'GLPI', 'GLQ', 'GLRE', 'GLSI', 'GLT', 'GLTO', 'GLUU', 'GLW', 'GLYC', 'GM', 'GMAB', 'GMDA', 'GME', 'GMED', 'GMRE', 'GMS', 'GNE', 'GNFT', 'GNK', 'GNL', 'GNLN', 'GNMK', 'GNPX', 'GNRC', 'GNRS', 'GNSS', 'GNT', 'GNTX', 'GNTY', 'GNW', 'GO', 'GOAC', 'GOCO', 'GOEV', 'GOF', 'GOGL', 'GOGO', 'GOL', 'GOLD', 'GOLF', 'GOOD', 'GOOG', 'GOOGL', 'GOOS', 'GOSS', 'GP', 'GPC', 'GPI', 'GPK', 'GPM', 'GPMT', 'GPN', 'GPP', 'GPRE', 'GPRK', 'GPRO', 'GPS', 'GPX', 'GRA', 'GRAY', 'GRBK', 'GRC', 'GRFS', 'GRIF', 'GRMN', 'GRPN', 'GRSV', 'GRTS', 'GRTX', 'GRUB', 'GRVY', 'GRWG', 'GRX', 'GS', 'GSAH', 'GSBC', 'GSBD', 'GSHD', 'GSIT', 'GSK', 'GSKY', 'GSL', 'GSS', 'GSX', 'GT', 'GTES', 'GTH', 'GTHX', 'GTLS', 'GTN', 'GTS', 'GTT', 'GTY', 'GTYH', 'GUT', 'GVA', 'GWAC', 'GWB', 'GWGH', 'GWPH', 'GWRE', 'GWRS', 'GWW', 'GXGX', 'H', 'HA', 'HAE', 'HAFC', 'HAIN', 'HAL', 'HALO', 'HARP', 'HAS', 'HASI', 'HAYN', 'HBAN', 'HBB', 'HBCP', 'HBI', 'HBIO', 'HBM', 'HBMD', 'HBNC', 'HBT', 'HCA', 'HCAT', 'HCC', 'HCCI', 'HCHC', 'HCI', 'HCKT', 'HCM', 'HCSG', 'HD', 'HDB', 'HE', 'HEAR', 'HEC', 'HEES', 'HEI', 'HELE', 'HEP', 'HEQ', 'HES', 'HESM', 'HEXO', 'HFC', 'HFFG', 'HFRO', 'HFWA', 'HGEN', 'HGLB', 'HGV', 'HHC', 'HHR', 'HI', 'HIBB', 'HIE', 'HIFS', 'HIG', 'HII', 'HIMX', 'HIO', 'HIW', 'HIX', 'HKIB', 'HL', 'HLF', 'HLG', 'HLI', 'HLIO', 'HLIT', 'HLNE', 'HLT', 'HLX', 'HLXA', 'HMC', 'HMHC', 'HMI', 'HMLP', 'HMN', 'HMST', 'HMSY', 'HMTV', 'HMY', 'HNGR', 'HNI', 'HNP', 'HNW', 'HOFT', 'HOG', 'HOL', 'HOLI', 'HOLX', 'HOMB', 'HOME', 'HON', 'HONE', 'HOOK', 'HOPE', 'HOV', 'HP', 'HPE', 'HPF', 'HPI', 'HPK', 'HPP', 'HPQ', 'HPS', 'HPX', 'HQH', 'HQI', 'HQL', 'HQY', 'HR', 'HRB', 'HRC', 'HRI', 'HRL', 'HRMY', 'HROW', 'HRTG', 'HRTX', 'HRZN', 'HSAQ', 'HSBC', 'HSC', 'HSIC', 'HSII', 'HSKA', 'HST', 'HSTM', 'HSY', 'HT', 'HTA', 'HTBI', 'HTBK', 'HTBX', 'HTD', 'HTGC', 'HTH', 'HTHT', 'HTLD', 'HTLF', 'HTOO', 'HUBB', 'HUBG', 'HUBS', 'HUIZ', 'HUM', 'HUN', 'HURC', 'HURN', 'HUYA', 'HVT', 'HWBK', 'HWC', 'HWKN', 'HWM', 'HXL', 'HY', 'HYB', 'HYFM', 'HYI', 'HYLN', 'HYMC', 'HYRE', 'HYT', 'HZAC', 'HZN', 'HZNP', 'HZO', 'HZON', 'IAA', 'IAC', 'IACA', 'IAE', 'IAF', 'IAG', 'IART', 'IBA', 'IBCP', 'IBEX', 'IBKR', 'IBM', 'IBN', 'IBOC', 'IBP', 'IBTX', 'ICAD', 'ICBK', 'ICE', 'ICFI', 'ICHR', 'ICL', 'ICLK', 'ICLR', 'ICPT', 'ICUI', 'IDA', 'IDCC', 'IDE', 'IDN', 'IDRA', 'IDT', 'IDXX', 'IDYA', 'IEA', 'IEC', 'IEP', 'IESC', 'IEX', 'IFF', 'IFN', 'IFRX', 'IFS', 'IGA', 'IGAC', 'IGD', 'IGI', 'IGIC', 'IGMS', 'IGR', 'IGT', 'IH', 'IHC', 'IHD', 'IHG', 'IHIT', 'IHRT', 'IIF', 'III', 'IIIN', 'IIIV', 'IIM', 'IIN', 'IIPR', 'IIVI', 'ILMN', 'ILPT', 'IMAB', 'IMAX', 'IMGN', 'IMKTA', 'IMMP', 'IMMR', 'IMNM', 'IMO', 'IMOS', 'IMPX', 'IMRA', 'IMTX', 'IMUX', 'IMV', 'IMVT', 'IMXI', 'INAQ', 'INBK', 'INBX', 'INCY', 'INDB', 'INFN', 'INFO', 'INFU', 'INFY', 'ING', 'INGN', 'INGR', 'INMB', 'INMD', 'INN', 'INO', 'INOD', 'INOV', 'INS', 'INSE', 'INSG', 'INSI', 'INSM', 'INSP', 'INSW', 'INT', 'INTC', 'INTU', 'INTZ', 'INVA', 'INVE', 'INVH', 'INZY', 'IONS', 'IOSP', 'IOVA', 'IP', 'IPA', 'IPAR', 'IPG', 'IPGP', 'IPHA', 'IPHI', 'IPI', 'IPOC', 'IPOD', 'IPOE', 'IPOF', 'IPV', 'IQ', 'IQI', 'IQV', 'IR', 'IRBT', 'IRCP', 'IRDM', 'IRM', 'IRMD', 'IRS', 'IRT', 'IRTC', 'IRWD', 'ISBC', 'ISD', 'ISEE', 'ISRG', 'ISSC', 'ISTR', 'IT', 'ITCB', 'ITCI', 'ITGR', 'ITI', 'ITIC', 'ITMR', 'ITOS', 'ITRG', 'ITRI', 'ITRN', 'ITT', 'ITUB', 'ITW', 'IVA', 'IVAC', 'IVC', 'IVH', 'IVR', 'IVZ', 'IX', 'J', 'JACK', 'JAMF', 'JAX', 'JAZZ', 'JBGS', 'JBHT', 'JBL', 'JBLU', 'JBSS', 'JBT', 'JCE', 'JCI', 'JCO', 'JCOM', 'JD', 'JDD', 'JE', 'JEF', 'JELD', 'JEMD', 'JEQ', 'JFIN', 'JFR', 'JG', 'JGH', 'JHB', 'JHG', 'JHI', 'JHS', 'JHX', 'JIH', 'JJSF', 'JKHY', 'JKS', 'JLL', 'JLS', 'JMIA', 'JNCE', 'JNJ', 'JNPR', 'JOBS', 'JOE', 'JOF', 'JOUT', 'JPC', 'JPI', 'JPM', 'JPS', 'JPT', 'JQC', 'JRI', 'JRO', 'JRS', 'JRVR', 'JSD', 'JTA', 'JTD', 'JWN', 'JWS', 'JYAC', 'JYNT', 'K', 'KAI', 'KALA', 'KALU', 'KALV', 'KAMN', 'KAR', 'KB', 'KBAL', 'KBH', 'KBR', 'KC', 'KDMN', 'KDNY', 'KDP', 'KE', 'KELYA', 'KELYB', 'KEN', 'KEP', 'KEX', 'KEY', 'KEYS', 'KF', 'KFRC', 'KFS', 'KFY', 'KGC', 'KHC', 'KIDS', 'KIM', 'KIN', 'KIO', 'KIRK', 'KKR', 'KL', 'KLAC', 'KLDO', 'KLIC', 'KLR', 'KMB', 'KMDA', 'KMF', 'KMI', 'KMPR', 'KMT', 'KMX', 'KN', 'KNDI', 'KNL', 'KNOP', 'KNSA', 'KNSL', 'KNTE', 'KNX', 'KO', 'KOD', 'KODK', 'KOF', 'KOP', 'KPTI', 'KR', 'KRA', 'KRC', 'KREF', 'KRG', 'KRMD', 'KRNT', 'KRNY', 'KRO', 'KRON', 'KROS', 'KRP', 'KRTX', 'KRUS', 'KRYS', 'KSM', 'KSMT', 'KSS', 'KSU', 'KT', 'KTB', 'KTF', 'KTOS', 'KURA', 'KVHI', 'KW', 'KWR', 'KXIN', 'KYMR', 'KYN', 'KZR', 'L', 'LAC', 'LAD', 'LADR', 'LAKE', 'LAMR', 'LANC', 'LAND', 'LARK', 'LASR', 'LATN', 'LAUR', 'LAWS', 'LAZ', 'LAZR', 'LAZY', 'LB', 'LBAI', 'LBC', 'LBRDA', 'LBRDK', 'LBRT', 'LBTYA', 'LBTYB', 'LBTYK', 'LC', 'LCAP', 'LCI', 'LCII', 'LCNB', 'LCUT', 'LCY', 'LDL', 'LDOS', 'LDP', 'LE', 'LEA', 'LEAF', 'LEAP', 'LECO', 'LEG', 'LEGH', 'LEGN', 'LEN', 'LEO', 'LESL', 'LEU', 'LEVI', 'LEVL', 'LFAC', 'LFC', 'LFTR', 'LFUS', 'LFVN', 'LGI', 'LGIH', 'LGND', 'LGVW', 'LH', 'LHCG', 'LHX', 'LI', 'LII', 'LILA', 'LILAK', 'LIN', 'LINC', 'LIND', 'LINX', 'LIQT', 'LITE', 'LIVK', 'LIVN', 'LIVX', 'LIZI', 'LJPC', 'LKFN', 'LKQ', 'LL', 'LLNW', 'LLY', 'LMAT', 'LMND', 'LMNL', 'LMNR', 'LMNX', 'LMPX', 'LMRK', 'LMT', 'LNC', 'LND', 'LNDC', 'LNG', 'LNN', 'LNT', 'LNTH', 'LOB', 'LOCO', 'LOGC', 'LOGI', 'LOMA', 'LOOP', 'LOPE', 'LORL', 'LOV', 'LOVE', 'LOW', 'LPG', 'LPI', 'LPL', 'LPLA', 'LPRO', 'LPSN', 'LPTH', 'LPX', 'LQDT', 'LRCX', 'LRMR', 'LRN', 'LSAQ', 'LSCC', 'LSF', 'LSI', 'LSPD', 'LSTR', 'LSXMA', 'LSXMB', 'LSXMK', 'LTC', 'LTHM', 'LTRN', 'LTRPA', 'LTRPB', 'LTRX', 'LU', 'LULU', 'LUMN', 'LUMO', 'LUNA', 'LUNG', 'LUV', 'LUXA', 'LVS', 'LW', 'LX', 'LXEH', 'LXFR', 'LXP', 'LXRX', 'LYB', 'LYFT', 'LYRA', 'LYTS', 'LYV', 'LZB', 'M', 'MA', 'MAA', 'MAC', 'MACU', 'MAG', 'MAIN', 'MAN', 'MANH', 'MANT', 'MANU', 'MAR', 'MARA', 'MAS', 'MASI', 'MASS', 'MAT', 'MATW', 'MATX', 'MAV', 'MAX', 'MAXN', 'MAXR', 'MBCN', 'MBI', 'MBIN', 'MBIO', 'MBT', 'MBUU', 'MBWM', 'MC', 'MCA', 'MCB', 'MCBC', 'MCBS', 'MCD', 'MCFE', 'MCFT', 'MCHP', 'MCI', 'MCK', 'MCMJ', 'MCN', 'MCO', 'MCR', 'MCRB', 'MCRI', 'MCS', 'MCY', 'MD', 'MDB', 'MDC', 'MDGL', 'MDLA', 'MDLZ', 'MDNA', 'MDP', 'MDRX', 'MDT', 'MDU', 'MDVL', 'MDWD', 'MDWT', 'MDXG', 'MEC', 'MED', 'MEDP', 'MEG', 'MEI', 'MELI', 'MEN', 'MEOH', 'MERC', 'MESA', 'MESO', 'MET', 'MFA', 'MFC', 'MFGP', 'MFIN', 'MFL', 'MFM', 'MFNC', 'MFT', 'MG', 'MGA', 'MGEE', 'MGF', 'MGI', 'MGIC', 'MGLN', 'MGM', 'MGNI', 'MGNX', 'MGP', 'MGPI', 'MGRC', 'MGTA', 'MGTX', 'MGU', 'MGY', 'MHD', 'MHF', 'MHH', 'MHI', 'MHK', 'MHN', 'MHO', 'MIC', 'MIDD', 'MIG', 'MIK', 'MIME', 'MIN', 'MIRM', 'MIST', 'MITK', 'MIXT', 'MIY', 'MKC', 'MKL', 'MKSI', 'MKTX', 'MLAB', 'MLAC', 'MLCO', 'MLHR', 'MLI', 'MLM', 'MLP', 'MLR', 'MLVF', 'MMAC', 'MMC', 'MMD', 'MMI', 'MMM', 'MMP', 'MMS', 'MMSI', 'MMT', 'MMU', 'MMX', 'MMYT', 'MN', 'MNKD', 'MNOV', 'MNP', 'MNR', 'MNRL', 'MNRO', 'MNSB', 'MNSO', 'MNST', 'MNTX', 'MO', 'MOD', 'MODN', 'MOFG', 'MOGO', 'MOH', 'MOMO', 'MOR', 'MORF', 'MORN', 'MOS', 'MOTN', 'MOV', 'MP', 'MPA', 'MPAA', 'MPB', 'MPC', 'MPLN', 'MPLX', 'MPV', 'MPW', 'MPWR', 'MPX', 'MQT', 'MQY', 'MRBK', 'MRC', 'MRCC', 'MRCY', 'MREO', 'MRK', 'MRLN', 'MRNA', 'MRNS', 'MRO', 'MRSN', 'MRTN', 'MRTX', 'MRUS', 'MRVI', 'MRVL', 'MS', 'MSA', 'MSB', 'MSBI', 'MSC', 'MSCI', 'MSD', 'MSEX', 'MSFT', 'MSGE', 'MSGN', 'MSGS', 'MSI', 'MSM', 'MSON', 'MSP', 'MSTR', 'MT', 'MTA', 'MTB', 'MTBC', 'MTCH', 'MTCR', 'MTD', 'MTDR', 'MTEM', 'MTG', 'MTH', 'MTLS', 'MTN', 'MTOR', 'MTRN', 'MTRX', 'MTSC', 'MTSI', 'MTT', 'MTW', 'MTX', 'MTZ', 'MU', 'MUA', 'MUC', 'MUE', 'MUFG', 'MUH', 'MUI', 'MUJ', 'MUR', 'MUS', 'MUSA', 'MVBF', 'MVF', 'MVIS', 'MVT', 'MWA', 'MWK', 'MX', 'MXF', 'MXIM', 'MXL', 'MYC', 'MYD', 'MYE', 'MYF', 'MYFW', 'MYGN', 'MYI', 'MYJ', 'MYN', 'MYOV', 'MYRG', 'NAC', 'NAD', 'NAN', 'NARI', 'NATH', 'NATI', 'NATR', 'NAV', 'NAVI', 'NAZ', 'NBA', 'NBAC', 'NBB', 'NBH', 'NBHC', 'NBIX', 'NBLX', 'NBN', 'NBR', 'NBSE', 'NBTB', 'NBTX', 'NC', 'NCA', 'NCBS', 'NCLH', 'NCMI', 'NCNA', 'NCNO', 'NCR', 'NCV', 'NCZ', 'NDAQ', 'NDLS', 'NDSN', 'NEA', 'NEE', 'NEM', 'NEN', 'NEO', 'NEOG', 'NEP', 'NESR', 'NET', 'NEU', 'NEV', 'NEW', 'NEWR', 'NEWT', 'NEX', 'NEXA', 'NFBK', 'NFE', 'NFG', 'NFH', 'NFJ', 'NFLX', 'NG', 'NGA', 'NGG', 'NGHC', 'NGM', 'NGMS', 'NGS', 'NGVC', 'NGVT', 'NH', 'NHC', 'NHF', 'NHI', 'NHIC', 'NHS', 'NI', 'NICE', 'NID', 'NIE', 'NIM', 'NIO', 'NIQ', 'NISN', 'NIU', 'NJR', 'NK', 'NKE', 'NKG', 'NKLA', 'NKSH', 'NKTR', 'NKTX', 'NKX', 'NL', 'NLOK', 'NLS', 'NLSN', 'NLTX', 'NLY', 'NMCI', 'NMFC', 'NMI', 'NMIH', 'NML', 'NMM', 'NMMC', 'NMR', 'NMRK', 'NMT', 'NMY', 'NMZ', 'NNBR', 'NNDM', 'NNI', 'NNN', 'NNOX', 'NNY', 'NOA', 'NOAH', 'NOC', 'NODK', 'NOG', 'NOK', 'NOMD', 'NOV', 'NOVA', 'NOVS', 'NOVT', 'NOW', 'NP', 'NPA', 'NPK', 'NPO', 'NPTN', 'NPV', 'NQP', 'NRC', 'NRG', 'NRGX', 'NRIM', 'NRIX', 'NRK', 'NRO', 'NRP', 'NRZ', 'NS', 'NSA', 'NSC', 'NSCO', 'NSH', 'NSIT', 'NSL', 'NSP', 'NSSC', 'NSTG', 'NTAP', 'NTB', 'NTCO', 'NTCT', 'NTES', 'NTG', 'NTGR', 'NTLA', 'NTNX', 'NTP', 'NTR', 'NTRA', 'NTRS', 'NTST', 'NTUS', 'NTZ', 'NUAN', 'NUE', 'NUM', 'NUO', 'NUS', 'NUV', 'NUVA', 'NUW', 'NUZE', 'NVAX', 'NVCR', 'NVDA', 'NVEC', 'NVEE', 'NVG', 'NVGS', 'NVMI', 'NVO', 'NVR', 'NVRO', 'NVS', 'NVST', 'NVT', 'NVTA', 'NWBI', 'NWE', 'NWFL', 'NWG', 'NWL', 'NWLI', 'NWN', 'NWPX', 'NWS', 'NWSA', 'NX', 'NXC', 'NXGN', 'NXJ', 'NXP', 'NXPI', 'NXQ', 'NXR', 'NXRT', 'NXST', 'NXTC', 'NYC', 'NYCB', 'NYMT', 'NYT', 'NZF', 'O', 'OAC', 'OACB', 'OAS', 'OBCI', 'OBNK', 'OC', 'OCFC', 'OCFT', 'OCN', 'OCSI', 'OCSL', 'OCUL', 'ODC', 'ODFL', 'ODP', 'ODT', 'OEC', 'OESX', 'OFC', 'OFED', 'OFG', 'OFIX', 'OFLX', 'OGE', 'OGS', 'OHI', 'OI', 'OIA', 'OII', 'OIIM', 'OIS', 'OKE', 'OKTA', 'OLED', 'OLLI', 'OLMA', 'OLN', 'OLP', 'OM', 'OMAB', 'OMC', 'OMCL', 'OMER', 'OMF', 'OMI', 'OMP', 'ON', 'ONB', 'ONCR', 'ONCS', 'ONCT', 'ONDS', 'ONE', 'ONEM', 'ONEW', 'ONTO', 'OOMA', 'OPBK', 'OPCH', 'OPEN', 'OPI', 'OPK', 'OPP', 'OPRA', 'OPRT', 'OPRX', 'OPT', 'OPTN', 'OPY', 'OR', 'ORA', 'ORAN', 'ORBC', 'ORC', 'ORCC', 'ORCL', 'ORGO', 'ORI', 'ORIC', 'ORLA', 'ORLY', 'ORMP', 'ORN', 'ORPH', 'ORRF', 'ORTX', 'OSB', 'OSBC', 'OSH', 'OSIS', 'OSK', 'OSMT', 'OSPN', 'OSTK', 'OSUR', 'OSW', 'OTEX', 'OTIC', 'OTIS', 'OTRA', 'OTRK', 'OTTR', 'OUT', 'OVBC', 'OVLY', 'OVV', 'OXFD', 'OXLC', 'OXM', 'OXSQ', 'OXY', 'OYST', 'OZK', 'OZON', 'PAA', 'PAAS', 'PAC', 'PACB', 'PACE', 'PACK', 'PACW', 'PAE', 'PAG', 'PAGP', 'PAGS', 'PAHC', 'PAI', 'PAM', 'PANA', 'PAND', 'PANW', 'PAR', 'PARR', 'PASG', 'PATK', 'PAYA', 'PAYC', 'PAYS', 'PAYX', 'PB', 'PBA', 'PBCT', 'PBF', 'PBFS', 'PBFX', 'PBH', 'PBI', 'PBIP', 'PBPB', 'PBR', 'PBT', 'PBY', 'PBYI', 'PCAR', 'PCB', 'PCG', 'PCH', 'PCI', 'PCK', 'PCM', 'PCN', 'PCOM', 'PCPL', 'PCQ', 'PCRX', 'PCSB', 'PCTI', 'PCTY', 'PCVX', 'PCYO', 'PD', 'PDAC', 'PDCE', 'PDCO', 'PDD', 'PDEX', 'PDFS', 'PDI', 'PDLB', 'PDM', 'PDS', 'PDT', 'PE', 'PEAK', 'PEB', 'PEBK', 'PEBO', 'PEG', 'PEGA', 'PEIX', 'PEN', 'PENN', 'PEO', 'PEP', 'PERI', 'PETQ', 'PETS', 'PFBC', 'PFBI', 'PFC', 'PFD', 'PFE', 'PFG', 'PFGC', 'PFHD', 'PFIS', 'PFL', 'PFLT', 'PFN', 'PFO', 'PFPT', 'PFS', 'PFSI', 'PFSW', 'PG', 'PGC', 'PGEN', 'PGNY', 'PGP', 'PGR', 'PGRE', 'PGTI', 'PH', 'PHAR', 'PHAT', 'PHD', 'PHG', 'PHGE', 'PHI', 'PHK', 'PHM', 'PHR', 'PHT', 'PI', 'PIAI', 'PICO', 'PII', 'PIM', 'PINC', 'PINE', 'PING', 'PINS', 'PIPR', 'PJT', 'PK', 'PKBK', 'PKE', 'PKG', 'PKI', 'PKO', 'PKOH', 'PKX', 'PLAB', 'PLAN', 'PLAY', 'PLBC', 'PLCE', 'PLD', 'PLG', 'PLL', 'PLM', 'PLMR', 'PLNT', 'PLOW', 'PLPC', 'PLRX', 'PLSE', 'PLT', 'PLTR', 'PLUG', 'PLUS', 'PLX', 'PLXS', 'PLYA', 'PLYM', 'PM', 'PMBC', 'PMF', 'PML', 'PMM', 'PMO', 'PMT', 'PMVC', 'PMVP', 'PMX', 'PNC', 'PNFP', 'PNI', 'PNM', 'PNNT', 'PNR', 'PNTG', 'PNW', 'PODD', 'POOL', 'POR', 'POST', 'POWI', 'POWL', 'POWW', 'PPBI', 'PPC', 'PPD', 'PPG', 'PPL', 'PPR', 'PPT', 'PQG', 'PRA', 'PRAA', 'PRAH', 'PRAX', 'PRCH', 'PRDO', 'PRFT', 'PRG', 'PRGO', 'PRGS', 'PRGX', 'PRI', 'PRIM', 'PRK', 'PRLB', 'PRLD', 'PRMW', 'PRO', 'PROF', 'PROG', 'PROS', 'PROV', 'PRPB', 'PRPH', 'PRPL', 'PRQR', 'PRS', 'PRSC', 'PRSP', 'PRTA', 'PRTC', 'PRTH', 'PRTK', 'PRTS', 'PRTY', 'PRU', 'PRVB', 'PRVL', 'PS', 'PSA', 'PSAC', 'PSB', 'PSEC', 'PSF', 'PSMT', 'PSN', 'PSNL', 'PSO', 'PSTG', 'PSTH', 'PSTI', 'PSTL', 'PSTX', 'PSX', 'PSXP', 'PTC', 'PTCT', 'PTEN', 'PTGX', 'PTK', 'PTNR', 'PTON', 'PTR', 'PTRS', 'PTSI', 'PTVCA', 'PTVCB', 'PTVE', 'PTY', 'PUBM', 'PUK', 'PUMP', 'PUYI', 'PVAC', 'PVBC', 'PVG', 'PVH', 'PWFL', 'PWOD', 'PWR', 'PXD', 'PYPD', 'PYPL', 'PZC', 'PZN', 'PZZA', 'QADA', 'QADB', 'QCOM', 'QCRH', 'QDEL', 'QELL', 'QFIN', 'QGEN', 'QH', 'QIWI', 'QLYS', 'QMCO', 'QNST', 'QQQX', 'QRTEA', 'QRTEB', 'QRVO', 'QS', 'QSR', 'QTNT', 'QTRX', 'QTS', 'QTWO', 'QUAD', 'QUMU', 'QUOT', 'QURE', 'R', 'RA', 'RACA', 'RACE', 'RAD', 'RADA', 'RADI', 'RAMP', 'RAPT', 'RARE', 'RAVN', 'RBA', 'RBAC', 'RBB', 'RBBN', 'RBC', 'RBCAA', 'RBNC', 'RC', 'RCEL', 'RCHG', 'RCI', 'RCII', 'RCKT', 'RCKY', 'RCL', 'RCM', 'RCS', 'RCUS', 'RDCM', 'RDFN', 'RDHL', 'RDI', 'RDIB', 'RDN', 'RDNT', 'RDUS', 'RDVT', 'RDWR', 'RDY', 'RE', 'REAL', 'REDU', 'REG', 'REGI', 'REGN', 'REKR', 'RELX', 'RENN', 'REPL', 'RES', 'RESI', 'RETA', 'REV', 'REVG', 'REX', 'REXR', 'REYN', 'REZI', 'RF', 'RFI', 'RFL', 'RFP', 'RGA', 'RGCO', 'RGEN', 'RGLD', 'RGNX', 'RGP', 'RGR', 'RGS', 'RGT', 'RH', 'RHI', 'RHP', 'RICE', 'RICK', 'RIDE', 'RIGL', 'RILY', 'RIO', 'RIOT', 'RIV', 'RJF', 'RKT', 'RL', 'RLAY', 'RLGT', 'RLGY', 'RLI', 'RLJ', 'RLMD', 'RM', 'RMAX', 'RMBI', 'RMBS', 'RMD', 'RMI', 'RMM', 'RMNI', 'RMR', 'RMRM', 'RMT', 'RNA', 'RNDB', 'RNET', 'RNG', 'RNLX', 'RNP', 'RNR', 'RNST', 'ROAD', 'ROCH', 'ROCK', 'ROG', 'ROIC', 'ROK', 'ROKU', 'ROL', 'ROLL', 'ROOT', 'ROP', 'ROST', 'RP', 'RPAI', 'RPAY', 'RPD', 'RPLA', 'RPM', 'RPRX', 'RPT', 'RPTX', 'RQI', 'RRBI', 'RRC', 'RRGB', 'RRR', 'RS', 'RSG', 'RTLR', 'RTP', 'RTX', 'RUBY', 'RUN', 'RUSHA', 'RUSHB', 'RUTH', 'RVI', 'RVLV', 'RVMD', 'RVNC', 'RVP', 'RVSB', 'RVT', 'RWT', 'RXN', 'RXT', 'RY', 'RYAAY', 'RYAM', 'RYI', 'RYN', 'RYTM', 'RZLT', 'SA', 'SABR', 'SAFE', 'SAFM', 'SAFT', 'SAGE', 'SAH', 'SAIA', 'SAIC', 'SAII', 'SAIL', 'SAL', 'SALT', 'SAM', 'SAMG', 'SAN', 'SAND', 'SANM', 'SAP', 'SAR', 'SASR', 'SATS', 'SAVA', 'SAVE', 'SBAC', 'SBCF', 'SBE', 'SBFG', 'SBG', 'SBGI', 'SBH', 'SBI', 'SBLK', 'SBNY', 'SBR', 'SBRA', 'SBS', 'SBSI', 'SBSW', 'SBT', 'SBTX', 'SBUX', 'SC', 'SCCO', 'SCD', 'SCHL', 'SCHN', 'SCHW', 'SCI', 'SCL', 'SCM', 'SCPE', 'SCPH', 'SCPL', 'SCS', 'SCSC', 'SCU', 'SCVL', 'SCVX', 'SCWX', 'SCYX', 'SD', 'SDC', 'SDGR', 'SE', 'SEAH', 'SEAS', 'SEB', 'SEDG', 'SEE', 'SEER', 'SEIC', 'SELB', 'SEM', 'SENEA', 'SENEB', 'SF', 'SFBS', 'SFE', 'SFIX', 'SFL', 'SFM', 'SFNC', 'SFST', 'SFT', 'SFTW', 'SFUN', 'SG', 'SGA', 'SGAM', 'SGC', 'SGEN', 'SGH', 'SGMO', 'SGMS', 'SGRY', 'SGTX', 'SGU', 'SHAK', 'SHBI', 'SHC', 'SHEN', 'SHG', 'SHI', 'SHLX', 'SHO', 'SHOO', 'SHOP', 'SHSP', 'SHW', 'SHYF', 'SI', 'SIBN', 'SIC', 'SID', 'SIEB', 'SIEN', 'SIG', 'SIGA', 'SIGI', 'SII', 'SILC', 'SILK', 'SILV', 'SIM', 'SIMO', 'SINA', 'SIRI', 'SITC', 'SITE', 'SITM', 'SIVB', 'SIX', 'SJ', 'SJI', 'SJM', 'SJR', 'SJW', 'SKLZ', 'SKM', 'SKT', 'SKX', 'SKY', 'SKYW', 'SLAB', 'SLB', 'SLCA', 'SLCT', 'SLDB', 'SLF', 'SLG', 'SLGL', 'SLGN', 'SLM', 'SLN', 'SLP', 'SLQT', 'SLRC', 'SM', 'SMAR', 'SMBC', 'SMBK', 'SMCI', 'SMED', 'SMFG', 'SMG', 'SMMC', 'SMMCU', 'SMMF', 'SMMT', 'SMP', 'SMPL', 'SMSI', 'SMTC', 'SMTS', 'SMTX', 'SNA', 'SNAP', 'SNBR', 'SNCR', 'SNDR', 'SNDX', 'SNE', 'SNEX', 'SNFCA', 'SNN', 'SNOW', 'SNP', 'SNPR', 'SNPS', 'SNR', 'SNV', 'SNX', 'SNY', 'SO', 'SOAC', 'SOGO', 'SOHU', 'SOI', 'SOL', 'SOLO', 'SOLY', 'SON', 'SONA', 'SONO', 'SOR', 'SP', 'SPB', 'SPCE', 'SPE', 'SPFI', 'SPG', 'SPGI', 'SPH', 'SPI', 'SPKE', 'SPLK', 'SPLP', 'SPNE', 'SPNS', 'SPNV', 'SPOK', 'SPOT', 'SPPI', 'SPR', 'SPRB', 'SPRO', 'SPSC', 'SPT', 'SPTN', 'SPWH', 'SPWR', 'SPXC', 'SPXX', 'SQ', 'SQM', 'SQNS', 'SQZ', 'SR', 'SRAC', 'SRC', 'SRCE', 'SRCL', 'SRDX', 'SRE', 'SRG', 'SRI', 'SRLP', 'SRNE', 'SRPT', 'SRRA', 'SRRK', 'SRSA', 'SRT', 'SSB', 'SSD', 'SSL', 'SSNC', 'SSP', 'SSPK', 'SSRM', 'SSSS', 'SSTI', 'SSTK', 'SSYS', 'ST', 'STAA', 'STAG', 'STAR', 'STAY', 'STBA', 'STC', 'STE', 'STEP', 'STFC', 'STIC', 'STIM', 'STK', 'STKL', 'STKS', 'STL', 'STLD', 'STM', 'STMP', 'STN', 'STND', 'STNE', 'STNG', 'STOK', 'STOR', 'STPK', 'STRA', 'STRL', 'STRO', 'STRS', 'STRT', 'STT', 'STTK', 'STWD', 'STWO', 'STX', 'STXB', 'STXS', 'STZ', 'SU', 'SUI', 'SUM', 'SUMO', 'SUN', 'SUNS', 'SUP', 'SUPN', 'SURF', 'SUZ', 'SV', 'SVA', 'SVAC', 'SVC', 'SVM', 'SVMK', 'SWAV', 'SWBI', 'SWCH', 'SWI', 'SWIR', 'SWK', 'SWKH', 'SWKS', 'SWM', 'SWTX', 'SWX', 'SWZ', 'SXC', 'SXI', 'SXT', 'SY', 'SYBT', 'SYF', 'SYK', 'SYKE', 'SYNA', 'SYNH', 'SYRS', 'SYX', 'SYY', 'T', 'TA', 'TAC', 'TACO', 'TAK', 'TAL', 'TALO', 'TAP', 'TARA', 'TARO', 'TARS', 'TAST', 'TBBK', 'TBI', 'TBIO', 'TBK', 'TBNK', 'TBPH', 'TCBI', 'TCBK', 'TCDA', 'TCF', 'TCFC', 'TCI', 'TCMD', 'TCOM', 'TCON', 'TCP', 'TCPC', 'TCRR', 'TCS', 'TCX', 'TD', 'TDAC', 'TDC', 'TDF', 'TDG', 'TDOC', 'TDS', 'TDW', 'TDY', 'TEAF', 'TEAM', 'TECH', 'TECK', 'TEDU', 'TEF', 'TEI', 'TEKK', 'TEL', 'TELA', 'TEN', 'TENB', 'TEO', 'TER', 'TEVA', 'TEX', 'TFC', 'TFFP', 'TFII', 'TFSL', 'TFX', 'TG', 'TGH', 'TGI', 'TGLS', 'TGNA', 'TGP', 'TGS', 'TGT', 'TGTX', 'THBR', 'THC', 'THCA', 'THCB', 'THFF', 'THG', 'THO', 'THQ', 'THR', 'THRM', 'THRY', 'THS', 'THW', 'TIF', 'TIG', 'TIGO', 'TIGR', 'TILE', 'TIMB', 'TIPT', 'TISI', 'TITN', 'TJX', 'TKC', 'TKR', 'TLC', 'TLK', 'TLMD', 'TLND', 'TLRY', 'TLYS', 'TM', 'TMDX', 'TME', 'TMHC', 'TMO', 'TMP', 'TMPM', 'TMST', 'TMTS', 'TMUS', 'TMX', 'TNAV', 'TNC', 'TNDM', 'TNET', 'TNK', 'TNP', 'TOL', 'TOT', 'TOWN', 'TPB', 'TPC', 'TPCO', 'TPGY', 'TPH', 'TPIC', 'TPL', 'TPR', 'TPRE', 'TPTX', 'TPVG', 'TPX', 'TR', 'TRC', 'TREB', 'TREC', 'TREE', 'TREX', 'TRGP', 'TRHC', 'TRI', 'TRIL', 'TRIP', 'TRIT', 'TRMB', 'TRMD', 'TRMK', 'TRN', 'TRNO', 'TRNS', 'TROW', 'TROX', 'TRP', 'TRQ', 'TRS', 'TRST', 'TRTN', 'TRTX', 'TRU', 'TRUP', 'TRV', 'TS', 'TSBK', 'TSC', 'TSCAP', 'TSCO', 'TSE', 'TSEM', 'TSHA', 'TSI', 'TSLA', 'TSLX', 'TSM', 'TSN', 'TSQ', 'TT', 'TTC', 'TTCF', 'TTD', 'TTEC', 'TTEK', 'TTGT', 'TTM', 'TTMI', 'TTWO', 'TU', 'TUFN', 'TUP', 'TUSK', 'TV', 'TVTX', 'TVTY', 'TW', 'TWCT', 'TWI', 'TWIN', 'TWLO', 'TWN', 'TWND', 'TWNK', 'TWO', 'TWOU', 'TWST', 'TWTR', 'TX', 'TXG', 'TXN', 'TXRH', 'TXT', 'TY', 'TYG', 'TYL', 'TZOO', 'U', 'UA', 'UAA', 'UAL', 'UAN', 'UAVS', 'UBA', 'UBER', 'UBFO', 'UBP', 'UBS', 'UBSI', 'UBX', 'UCBI', 'UCL', 'UCTT', 'UDR', 'UE', 'UEIC', 'UEPS', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UFS', 'UGI', 'UGP', 'UHAL', 'UHS', 'UHT', 'UI', 'UIHC', 'UIS', 'UK', 'UL', 'ULBI', 'ULH', 'ULTA', 'UMBF', 'UMC', 'UMH', 'UMPQ', 'UNB', 'UNF', 'UNFI', 'UNH', 'UNIT', 'UNM', 'UNP', 'UNTY', 'UNVR', 'UONE', 'UPLD', 'UPS', 'UPST', 'UPWK', 'URBN', 'URGN', 'URI', 'UROV', 'USA', 'USAC', 'USAS', 'USAT', 'USB', 'USCR', 'USFD', 'USLM', 'USM', 'USNA', 'USPH', 'USX', 'UTF', 'UTG', 'UTHR', 'UTI', 'UTL', 'UTMD', 'UTZ', 'UUUU', 'UVE', 'UVSP', 'UVV', 'V', 'VAC', 'VACQ', 'VALE', 'VALU', 'VAPO', 'VAR', 'VBF', 'VBTX', 'VC', 'VCEL', 'VCIF', 'VCRA', 'VCTR', 'VCV', 'VCYT', 'VEC', 'VECO', 'VEDL', 'VEEV', 'VEL', 'VER', 'VERI', 'VERU', 'VERX', 'VERY', 'VET', 'VFC', 'VFF', 'VG', 'VGAC', 'VGI', 'VGM', 'VGR', 'VHC', 'VHI', 'VIAC', 'VIACA', 'VIAO', 'VIAV', 'VICI', 'VICR', 'VIE', 'VIH', 'VINC', 'VIOT', 'VIPS', 'VIR', 'VIRT', 'VITL', 'VIV', 'VIVO', 'VKI', 'VKQ', 'VKTX', 'VLDR', 'VLGEA', 'VLO', 'VLRS', 'VLY', 'VMAC', 'VMAR', 'VMC', 'VMD', 'VMI', 'VMM', 'VMO', 'VMW', 'VNDA', 'VNE', 'VNET', 'VNO', 'VNOM', 'VNRX', 'VNT', 'VNTR', 'VOD', 'VOXX', 'VOYA', 'VPG', 'VPV', 'VRA', 'VRAY', 'VRCA', 'VREX', 'VRM', 'VRNA', 'VRNS', 'VRNT', 'VRRM', 'VRS', 'VRSK', 'VRSN', 'VRT', 'VRTS', 'VRTU', 'VRTV', 'VRTX', 'VSAT', 'VSEC', 'VSH', 'VSPR', 'VST', 'VSTA', 'VSTO', 'VTA', 'VTN', 'VTOL', 'VTR', 'VTRS', 'VTRU', 'VUZI', 'VVI', 'VVNT', 'VVOS', 'VVPR', 'VVR', 'VVV', 'VXRT', 'VYGG', 'VYGR', 'VZ', 'W', 'WAB', 'WABC', 'WAFD', 'WAL', 'WASH', 'WAT', 'WB', 'WBA', 'WBAI', 'WBK', 'WBS', 'WBT', 'WCC', 'WCN', 'WD', 'WDAY', 'WDC', 'WDFC', 'WDR', 'WEA', 'WEC', 'WELL', 'WEN', 'WERN', 'WES', 'WETF', 'WEX', 'WEYS', 'WF', 'WFC', 'WGO', 'WH', 'WHD', 'WHF', 'WHG', 'WHR', 'WIA', 'WIFI', 'WILC', 'WIMI', 'WINA', 'WING', 'WIRE', 'WISH', 'WIT', 'WIW', 'WIX', 'WK', 'WKEY', 'WKHS', 'WLDN', 'WLFC', 'WLK', 'WLKP', 'WLL', 'WLTW', 'WM', 'WMB', 'WMC', 'WMG', 'WMK', 'WMS', 'WMT', 'WNC', 'WNEB', 'WNS', 'WNW', 'WOR', 'WORK', 'WOW', 'WPC', 'WPF', 'WPG', 'WPM', 'WPP', 'WPRT', 'WPX', 'WRAP', 'WRB', 'WRE', 'WRI', 'WRK', 'WRLD', 'WSBC', 'WSBF', 'WSC', 'WSFS', 'WSM', 'WSO', 'WSR', 'WST', 'WTBA', 'WTFC', 'WTM', 'WTRE', 'WTRG', 'WTS', 'WTTR', 'WU', 'WVE', 'WW', 'WWD', 'WWE', 'WWW', 'WY', 'WYND', 'WYNN', 'X', 'XAN', 'XBIT', 'XEC', 'XEL', 'XENE', 'XENT', 'XERS', 'XFOR', 'XGN', 'XHR', 'XL', 'XLNX', 'XLRN', 'XNCR', 'XOM', 'XOMA', 'XONE', 'XP', 'XPEL', 'XPER', 'XPEV', 'XPO', 'XPOA', 'XRAY', 'XRX', 'XYL', 'Y', 'YAC', 'YALA', 'YELP', 'YETI', 'YEXT', 'YGMZ', 'YI', 'YMAB', 'YMTX', 'YNDX', 'YORW', 'YPF', 'YQ', 'YRCW', 'YRD', 'YSAC', 'YSG', 'YUM', 'YUMC', 'YY', 'Z', 'ZAGG', 'ZBH', 'ZBRA', 'ZEAL', 'ZEN', 'ZEUS', 'ZG', 'ZGNX', 'ZI', 'ZION', 'ZIXI', 'ZLAB', 'ZM', 'ZNGA', 'ZNH', 'ZNTL', 'ZS', 'ZTO', 'ZTR', 'ZTS', 'ZUMZ', 'ZUO', 'ZVO', 'ZYME', 'ZYXI', 'TRUE'}


# In[6]:


# words which seem like stock symbols but are regular words
blacklist = {'I', 'ARE',  'ON', 'GO', 'NOW', 'CAN', 'UK', 'SO', 'OR', 'OUT', 'SEE', 'ONE', 'LOVE', 'U', 'STAY', 'HAS', 'BY', 'BIG', 'GOOD', 'RIDE', 'EOD', 'ELON', 'WSB', 'THE', 'A', 'ROPE', 'YOLO', 'TOS', 'CEO', 'DD', 'IT', 'OPEN', 'ATH', 'PM', 'IRS', 'FOR','DEC', 'BE', 'IMO', 'ALL', 'RH', 'EV', 'TOS', 'CFO', 'CTO', 'DD', 'BTFD', 'WSB', 'OK', 'PDT', 'RH', 'KYS', 'FD', 'TYS', 'US', 'USA', 'IT', 'ATH', 'RIP', 'BMW', 'GDP', 'OTM', 'ATM', 'ITM', 'IMO', 'LOL', 'AM', 'BE', 'PR', 'PRAY', 'PT', 'FBI', 'SEC', 'GOD', 'NOT', 'POS', 'FOMO', 'TL;DR', 'EDIT', 'STILL', 'WTF', 'RAW', 'PM', 'LMAO', 'LMFAO', 'ROFL', 'EZ', 'RED', 'BEZOS', 'TICK', 'IS', 'PM', 'LPT', 'GOAT', 'FL', 'CA', 'IL', 'MACD', 'HQ', 'OP', 'PS', 'AH', 'TL', 'JAN', 'FEB', 'JUL', 'AUG', 'SEP', 'SEPT', 'OCT', 'NOV', 'FDA', 'IV', 'ER', 'IPO', 'MILF', 'BUT', 'SSN', 'FIFA', 'USD', 'CPU', 'AT', 'GG', 'Mar'}


# In[7]:


# reddit stock domain words
domain_words = {
    'citron': -4.0,  
    'hidenburg': -4.0,        
    'moon': 4.0,
    'highs': 2.0,
    'mooning': 4.0,
    'long': 2.0,
    'short': -2.0,
    'call': 4.0,
    'calls': 4.0,    
    'put': -4.0,
    'puts': -4.0,    
    'break': 2.0,
    'tendie': 2.0,
     'tendies': 2.0,
     'town': 2.0,     
     'overvalued': -3.0,
     'undervalued': 3.0,
     'buy': 4.0,
     'sell': -4.0,
     'gone': -1.0,
     'gtfo': -1.7,
     'paper': -1.7,
     'bullish': 3.7,
     'bearish': -3.7,
     'bagholder': -1.7,
     'stonk': 1.9,
     'green': 1.9,
     'money': 1.2,
     'print': 2.2,
     'rocket': 2.2,
     'bull': 2.9,
     'bear': -2.9,
     'pumping': -1.0,
     'sus': -3.0,
     'offering': -2.3,
     'rip': -4.0,
     'downgrade': -3.0,
     'upgrade': 3.0,     
     'maintain': 1.0,          
     'pump': 1.9,
     'hot': 1.5,
     'drop': -2.5,
     'rebound': 1.5,  
     'crack': 2.5,}


# ### Extracting tickers from Submissions and Comments

# In[8]:


tickers, a_text = {}, {}
picks = 100 # top ticks to pick 
text_frames = [submissions["text"], comments["text"]]
texts = pd.concat(text_frames)


# In[9]:


for text in texts:
    split = str(text).split(" ")
    for word in split:
        word = word.replace("$", "")        
        # upper = ticker, length of ticker <= 5, excluded words,                     
        if word.isupper() and len(word) <= 5 and word not in blacklist and word in US_STOCK_SYMBOLS:
            if word in tickers:
                tickers[word] += 1
                a_text[word].append(text)
            else:
                a_text[word] = [text]
                tickers[word] = 1


# In[10]:


# sorting the dictionary and picking the top tickers

symbols = dict(sorted(tickers.items(), key=lambda item: item[1], reverse = True))
top_picks = list(symbols.keys())[0:picks]


# ## Write ticker mentions to csv

# In[23]:


import csv
# field_names = ["Tickers","Mentions"]

with open('../../data/LabelledData/TickerMentions.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in symbols.items():
       writer.writerow([key, value])


# In[31]:


print(f"\n{picks} most mentioned picks: ")
times = []
top = []
for i in top_picks:
    print(f"{i}: {symbols[i]}")
    times.append(symbols[i])
    top.append(f"{i}: {symbols[i]}")
   


# ## Sentiment Analysis for these ticks

from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


# In[ ]:


import torch

classes = ["positive", "negative","neutral"]

# sequence_0 = "No they do not, but others only day trade options. If your going to day trade options, stick with In The Money (ITM) or Near The Money (NTM) strikes, with close expiration's, on highly liquid stocks or ETFs. I trade weekly Russel 2000 (IWM) options almost everyday."
# sequence_1 = "Stocks rallied and the British pound gained."
# sequence_2 = "TSLA is sacked going into shithole."
print(f"\nSentiment analysis of top {picks} picks with FinBERT")

picks_sentiment = list(symbols.keys())[0:picks]
# Applying Sentiment Analysis
scores, sentiment = {}, {}
count = 0
for symbol in picks_sentiment:
    stock_texts = a_text[symbol]
    count = count + 1
    print(f"\Symbols labelled {count} with FinBERT")
    for text in stock_texts:
#         print(count,text)
        inputs = tokenizer.encode_plus(text,add_special_tokens=True, truncation=True, return_tensors="pt")
        logits = model(**inputs)[0]
        result = torch.softmax(logits, dim=1).tolist()[0]
        score = {'neg': result[1], 'neu': result[2], 'pos': result[0]}
#         print(score)
        if symbol in sentiment:
            sentiment[symbol][text] = score
        else:
            sentiment[symbol] = {text:score}      
        if symbol in scores:
            for key, _ in score.items():
                scores[symbol][key] += score[key]
        else:
            scores[symbol] = score
            
    # calculating avg.
    for key in score:
        scores[symbol][key] = scores[symbol][key] / symbols[symbol]
        scores[symbol][key]  = "{pol:.3f}".format(pol=scores[symbol][key])
 
print(f"\nSentiment analysis done of top {picks} picks")
df = pd.DataFrame(scores)
df.index = ['Bearish', 'Neutral', 'Bullish']
df = df.T

df.to_csv("../../data/LabelledData/ExtractedTicks_Labelled_FinBERT.csv")
print("Results are written to file ../../data/LabelledData/ExtractedTicks_Labelled_FinBERT.csv")

