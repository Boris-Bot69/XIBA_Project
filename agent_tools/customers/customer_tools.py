import os
import csv
from typing import List
import pandas as pd
from langchain_core.tools import tool

from utils import get_cwd

@tool
def onboard_customer(company_name: str, name: str, domain: str, email: str, phone: str, thread_id: str, request_date: str, request_summary: str) -> List[dict]:
    """Onboard New Customer by creating a profile and identifying issues. phone is optional."""
    # create a csv with customer profile, match with email address. If already exists, do nothing
    try:
        profiles_path = get_cwd() + os.sep + 'data' + os.sep + 'customer_profiles.csv'
        columns = ['customer_id','company_name', 'name', 'domain', 'email', 'phone', 'thread_id', 'request_date', 'request_summary']
        print ("\n\nonboard_customer called with: ", company_name, name, domain, email, phone, thread_id, request_date, request_summary)
        create_profile = False
        profiles_df = None
        if os.path.exists(profiles_path):
            profiles_df = pd.read_csv(profiles_path)
            if profiles_df[profiles_df['email'] == email].empty:
                create_profile = True
        else:
            create_profile = True        
        if create_profile:
            try:
                profiles_suffix = f"{profiles_df.shape[0] + 1:03}"
            except Exception as e:
                profiles_suffix = "001"
            row_data = {
                'customer_id': f"CUST-{profiles_suffix}",
                'company_name': company_name,
                'name': name,
                'domain': domain,
                'email': email,
                'phone': phone if phone else '',
                'thread_id': thread_id,
                'request_date': request_date,
                'request_summary': request_summary
            }

            # CSV file path
            csv_file = profiles_path

            # Convert the dictionary to a single-row dictionary (take the first element from each list)
            single_row = {col: "%s" % row_data.get(col) for col in columns}

            # Check if file exists to write header only once
            file_exists = os.path.isfile(csv_file)

            # Append to CSV
            with open(csv_file, mode='a', newline='\n', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                
                if not file_exists:
                    writer.writeheader()  # Write header if file does not exist
                
                writer.writerow(single_row) 


                # profiles_df = pd.DataFrame({
                #     'customer_id': [f"CUST-{profiles_df.shape[0] + 1:03}"],
                #     'company_name': [company_name],
                #     'name': [name],
                #     'domain': [domain],
                #     'email': [email],
                #     'phone': [phone if phone else ''],
                #     'thread_id': [thread_id],
                #     'request_date': [request_date],
                #     'request_summary': [request_summary]
                # })

                # profiles_df.to_csv(profiles_path, mode='a', index=False, header=False)

                # profiles_df = pd.DataFrame(columns=['customer_id','company_name', 'name', 'domain', 'email', 'phone', 'thread_id', 'request_date', 'request_summary'])
                # profiles_df = profiles_df.concat({
                #     'customer_id': f"CUST-{profiles_df.shape[0] + 1:03}",
                #     'company_name': company_name,
                #     'name': name,
                #     'domain': domain,
                #     'email': email,
                #     'phone': phone if phone else '',
                #     'thread_id': thread_id,
                #     'request_date': request_date,
                #     'request_summary': request_summary
                # }, ignore_index=True)
                # profiles_df.to_csv(profiles_path, index=False)
                # profiles_df.save(profiles_path, index=False)
                print ("Customer profile created for ", name)
                return [{"message": f"Customer profile created for {name}", "customer_id": f"CUST-{profiles_suffix}"}]
        else:
            # import pdb; pdb.set_trace()
            cdf = profiles_df[profiles_df['email'] == email]
            print ("Customer profile already exists for %s--%s--" % (cdf.name.values[0], cdf.customer_id.values[0]))
            return [{"message": f"Customer profile already exists for {cdf.name.values[0]}", "customer_id": cdf.customer_id.values[0]}]
    except Exception as e:
        print (f"Error onboarding customer: {e}")
        return [{"error": f"Error onboarding customer: {str(e)}"}]