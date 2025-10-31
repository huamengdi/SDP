import javalang
import csv
import os
import pandas as pd

projects = [
    'ant',
]
versions = {
    'ant':['1.5', '1.6'],
}
dict_dir = {"ant": ["src\main"]}
def dfs_with_positions(file_path):

    with open(file_path, "r", encoding="utf-8") as fd:
        source_code = fd.read()

    treenode = javalang.parse.parse(source_code)

    words = ''
    positions = ''

    def dfs(node, depth=0, index=0):
        nonlocal words, positions

        if isinstance(node, javalang.tree.ReferenceType):
            words += f' ReferenceType_{str(node.name)}'
            positions += f' {(depth, index)}'
        elif isinstance(node, (javalang.tree.MethodInvocation, javalang.tree.SuperMethodInvocation)):
            words += f' MethodInvocation_{str(node.member)}'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.MethodDeclaration):
            words += f' MethodDeclaration_{str(node.name)}'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.TypeDeclaration):
            words += f' TypeDeclaration_{str(node.name)}'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ClassDeclaration):
            words += f' ClassDeclaration_{str(node.name)}'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.EnumDeclaration):
            words += f' EnumDeclaration_{str(node.name)}'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.IfStatement):
            words += ' if'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.WhileStatement):
            words += ' while'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.DoStatement):
            words += ' do'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ForStatement):
            words += ' for'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.AssertStatement):
            words += ' assert'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.BreakStatement):
            words += ' break'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ContinueStatement):
            words += ' continue'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ReturnStatement):
            words += ' return'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ThrowStatement):
            words += ' throw'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.SynchronizedStatement):
            words += ' synchronized'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.TryStatement):
            words += ' try'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.SwitchStatement):
            words += ' switch'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.SwitchStatementCase):
            words += ' case'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.BlockStatement):
            words += ' block'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.StatementExpression):
            words += ' statementexpression'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.TryResource):
            words += ' tryresource'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.CatchClause):
            words += ' catch'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.CatchClauseParameter):
            words += ' catchparameter'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ForControl):
            words += ' forcontrol'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.EnhancedForControl):
            words += ' enhancedforcontrol'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ClassCreator):
            words += ' classcreator'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.VariableDeclarator):
            words += ' variable'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.InterfaceDeclaration):
            words += ' interface'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.PackageDeclaration):
            words += ' package'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.FormalParameter):
            words += ' parameter'
            positions += f' {(depth, index)}'
        elif isinstance(node, javalang.tree.ConstructorDeclaration):
            words += ' constructor'
            positions += f' {(depth, index)}'

        children = getattr(node, 'children', [])
        for i, child in enumerate(children):
            if isinstance(child, (list, tuple)):
                for item in child:
                    if item is not None:
                        dfs(item, depth + 1, i)
            elif child is not None:
                dfs(child, depth + 1, i)

    dfs(treenode)

    return words, positions


for project in projects:
    for version in versions[project]:

        promise_data_path = r".your_promise_data_path_.csv"
        output_csv_path = r".your_output_csv_path.csv"
        directory = os.path.dirname(output_csv_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(["file_name", "bug","node","position","wmc", "dit", "noc", "cbo", 'rfc',
                                 "lcom", "ca", "ce", "npm","lcom3", "loc", "dam", "moa", "mfa",
                                 "cam", "ic", "cbm","amc","max_cc","avg_cc" ])  

        data = pd.read_csv(promise_data_path)


        if 'name' not in data.columns or 'bug' not in data.columns:
            print(f"'name' or 'bug' column is missing in {promise_data_path}")
            continue


        name_index = data.columns.get_loc('name') 
        bug_index = data.columns.get_loc('bug')  
        wmc_index = data.columns.get_loc('wmc')
        dit_index = data.columns.get_loc('dit')
        noc_index = data.columns.get_loc('noc')
        cbo_index = data.columns.get_loc('cbo')
        rfc_index = data.columns.get_loc('rfc')
        lcom_index = data.columns.get_loc('lcom')
        ca_index = data.columns.get_loc('ca')
        ce_index = data.columns.get_loc('ce')
        npm_index = data.columns.get_loc('npm')
        lcom3_index = data.columns.get_loc('lcom3')
        loc_index = data.columns.get_loc('loc')
        dam_index = data.columns.get_loc('dam')
        moa_index = data.columns.get_loc('moa')
        mfa_index = data.columns.get_loc('mfa')
        cam_index = data.columns.get_loc('cam')
        ic_index = data.columns.get_loc('ic')
        cbm_index = data.columns.get_loc('cbm')
        amc_index = data.columns.get_loc('amc')
        max_cc_index = data.columns.get_loc('max_cc')
        avg_cc_index = data.columns.get_loc('avg_cc')


        for index, line in data.iterrows():

            bug_status = line[bug_index]  

            wmc = line[wmc_index]
            dit = line[dit_index]
            noc = line[noc_index]
            cbo = line[cbo_index]
            rfc = line[rfc_index]
            lcom = line[lcom_index]
            ca = line[ca_index]
            ce = line[ce_index]
            npm = line[npm_index]
            lcom3 = line[lcom3_index]
            loc = line[loc_index]
            dam = line[dam_index]
            moa = line[moa_index]
            mfa = line[mfa_index]
            cam = line[cam_index]
            ic = line[ic_index]
            cbm = line[cbm_index]
            amc = line[amc_index]
            max_cc = line[max_cc_index]
            avg_cc = line[avg_cc_index]


            for pos_dir in dict_dir[project]:
                path = r".\data\{}\{}\{}\{}".format(project, version, version, pos_dir)
                file_name = line[name_index].replace(".", "\\").split("$")[0] + ".java"
                file_path = os.path.join(path, file_name)
                if os.path.exists(file_path):
                    break
            else:
                print(f'{file_name} is not found.')
                continue

            try:
                words, positions=dfs_with_positions(file_path)


                with open(output_csv_path, mode='a', newline='', encoding='utf-8') as output_file:
                    csv_writer = csv.writer(output_file)
                    csv_writer.writerow([
                        file_path, bug_status, words.strip(),positions.strip(),
                        wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam,
                        moa, mfa, cam, ic, cbm, amc, max_cc, avg_cc
                    ])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print("File tokens have been saved to", output_csv_path)
