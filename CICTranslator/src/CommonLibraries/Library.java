package CommonLibraries;

import hopes.cic.xml.*;

import java.io.*;
import java.util.ArrayList;
import java.util.Map;

import InnerDataStructures.*;

public class Library {
	
	public static void generateLibraryCode(String mDestFile, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm){
		File fileOut = new File(mDestFile + library.getName() + ".h");	
		generateLibraryHeaderfile(fileOut, library);
		
		/*
		fileOut = new File(mDestFile + "l_" + library.getName() + ".h");	
		generateLibraryHeaderfile(fileOut, library);
		*/
		
		fileOut = new File(mDestFile + library.getName() + ".c");
		generateLibraryCfile(fileOut, library, mAlgorithm, "Normal");
	}
	
	public static void generateLibraryWrapperCode(String mDestFile, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm){
		File fileOut = new File(mDestFile + library.getName() + ".h");	
		generateLibraryHeaderfile(fileOut, library);
		
		fileOut = new File(mDestFile + library.getName() + "_wrapper.c");
		generateLibraryWrapperFile(fileOut, library, mAlgorithm);
		
		fileOut = new File(mDestFile + library.getName() + "_data_structure.h");
		generateLibraryDataStructureHeader(fileOut, library);
	}
	

	public static void generateLibraryStubCode(String mDestFile, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm, boolean interrupt){		
		File fileOut = new File(mDestFile + library.getName() + ".h");	
		generateLibraryHeaderfile(fileOut, library);
		
		fileOut = new File(mDestFile + library.getName() + ".c");
		generateLibraryCfile(fileOut, library, mAlgorithm, "Stub");
		
		fileOut = new File(mDestFile + library.getName() + "_stub.c");
		generateLibraryStubFile(fileOut, 0, library, mAlgorithm, interrupt);
		
		fileOut = new File(mDestFile + library.getName() + "_data_structure.h");
		generateLibraryDataStructureHeader(fileOut, library);
	}
	
	public static void generateLibraryHeaderfile(File fileOut, InnerDataStructures.Library library){

		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(("#ifndef __HEADER_GUARD_" + library.getName() + "\n").getBytes());
			outstream.write(("#define __HEADER_GUARD_" + library.getName() + "\n\n").getBytes());
			outstream.write(("#define LIBFUNC(rtype, f, ...) rtype l_" + library.getName() + "_##f(__VA_ARGS__)\n\n").getBytes());
			outstream.write(("#include \"" + library.getHeader() + "\"\n").getBytes());
			outstream.write("\n#undef LIBFUNC\n#endif\n".getBytes());
			
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void generateLibraryCfile(File fileOut, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm, String type){
		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(("#ifndef __HEADER_GUARD__\n").getBytes());
			outstream.write(("#define __HEADER_GUARD__\n\n").getBytes());
			outstream.write("#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)\n".getBytes());
			outstream.write(("#define LIBCALL_this(f, ...) l_" + library.getName() + "_##f(__VA_ARGS__)\n").getBytes());
			if(mAlgorithm.getLibraryConnections() != null){
				for(LibraryLibraryConnectionType libLibCon: mAlgorithm.getLibraryConnections().getLibraryLibraryConnection()){
					if(libLibCon.getMasterLibrary().equals(library.getName())){
						outstream.write(("\n#include \""+ libLibCon.getSlaveLibrary() +".h\"\n").getBytes());
	                    outstream.write(("#define LIBCALL_"+ libLibCon.getMasterPort() +"(f, ...) l_"+ libLibCon.getSlaveLibrary() +"_##f(__VA_ARGS__)\n").getBytes());
					}
				}
			}
			
			outstream.write(("#define LIBFUNC(rtype, f, ...) rtype l_" + library.getName() + "_##f(__VA_ARGS__)\n\n").getBytes());
			if(type.equals("Normal"))		outstream.write(("#include \"" + library.getFile() + "\"\n").getBytes());
			else if(type.equals("Stub"))	outstream.write(("#include \"" + library.getName() + "_stub.c\"\n").getBytes());
			outstream.write("\n#undef LIBFUNC\n#endif\n".getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void generateLibraryStubFile(File fileOut, int taskLibraryFlag, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm, boolean interrupt)
	{

		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(translateLibraryStubFile(taskLibraryFlag, library, mAlgorithm, interrupt).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	public static String translateLibraryStubFile(int taskLibraryFlag, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm, boolean interrupt){
		int index = 0;
		String content = new String();	
	    
	    content += "#include \"LIB_port.h\"\n";
	    content += "#include \"target_system_model.h\"\n";
	    content += "#include \"target_task_model.h\"\n";
	    content += "#include \"" + library.getName() + "_data_structure.h\"\n\n";
	    content += "#define LIBFUNC(rtype, f, ...) rtype l_" + library.getName() + "_##f(__VA_ARGS__)\n\n";
	    //content += "static MUTEX_TYPE stub_mutex = MUTEX_INIT_INLINE;\n\n";
	    content += "static MUTEX_TYPE stub_mutex;\n\n";
	    content += "LIBFUNC(void, init, void)\n{\n\tMUTEX_INIT(&stub_mutex);\n\t// initialize\n}\nLIBFUNC(void, wrapup, void)\n{\n\tMUTEX_WRAPUP(&stub_mutex);\n\t// wrapup\n}\n";

	    for(Function func: library.getFuncList()){
	    	content += "LIBFUNC(" + func.getReturnType() + ", " + func.getFunctionName();
	    	int count = 0;
	    	for(Argument arg: func.getArgList()){
	    		content += ", " + arg.getType() + " " + arg.getVariableName();
	    		count++;
	    	}
	    	if(count == 0)	content += ", void";
	    	
	    	content += ")\n{\n";
	    	content += "\t" + library.getName() + "_func_data send_data;\n";
	    	if(!func.getReturnType().equals("void"))	content += "\t" + library.getName() + "_ret_data receive_data;\n\t";
	    	
	    	if(!func.getReturnType().equals("void"))	content += func.getReturnType() + " ret;\n\n";
	    	
	    	content += "\tint write_channel_id = init_lib_port(" + library.getIndex() + ", 'w');\n\n";
	    	if(!func.getReturnType().equals("void"))	
	    		content += "\tint read_channel_id = init_lib_port(" + library.getIndex() + ", 'r');\n\n";
	    	
	    	if(taskLibraryFlag == 1){
	    		content += "\tsend_data.task_library = 1;\n";
	    		content += "\tsend_data.task_id = get_mytask_id();\n";
	    	}
	    	else if(taskLibraryFlag == 2){
	    		content += "\tsend_data.task_library = 2;\n";
	    		content += "\tsend_data.task_id = " + library.getIndex() + ";\n";
	    	}
	    	content += "\tsend_data.func_num = " + func.getIndex() + ";\n"; 
	    	
	    	for(Argument arg: func.getArgList())
	    		content += "\tsend_data.func." + func.getFunctionName() + "." + arg.getVariableName() + " = " + arg.getVariableName() + ";\n";
	    	
	    	content += "\n\t// write port\n";
	    	//content += "\tlock_lib_channel(channel_id);\n";
	    	content += "\tMUTEX_LOCK(&stub_mutex);\n";
	    	content += "\tLIB_SEND(write_channel_id, &send_data, sizeof(" + library.getName() + "_func_data));\n";
    	
	    	if(!func.getReturnType().equals("void")){
	    		if(!interrupt){
			    	content += "\t// read port\n";
			    	content += "\twhile(1){\n";
			    	content += "\t\tif(LIB_CHECK(read_channel_id) >= sizeof(" + library.getName() + "_ret_data))	break;\n";
			    	content += "\t\telse SCHED_YIELD();\n\t}\n";
	    		}
		    	content += "\tLIB_RECEIVE(read_channel_id, &receive_data, sizeof(" + library.getName() + "_ret_data));\n";
		    	//content += "\tunlock_lib_channel(channel_id);\n";
		    	
	    		content += "\tret = receive_data.ret.ret_" + func.getFunctionName() + ";\n";
	    		content += "\tMUTEX_UNLOCK(&stub_mutex);\n\n";
	    		content += "\treturn ret;\n";
	    	}
	    	else	content += "\tMUTEX_UNLOCK(&stub_mutex);\n";
	    	
	    	content += "}\n\n";
	    }
	    content += "#undef LIBFUNC\n\n";
	    
		return content;
	}
	

	public static void generateLibraryWrapperFile(File fileOut, InnerDataStructures.Library library, CICAlgorithmType mAlgorithm)
	{
		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(translateLibraryWrapperFile(library, mAlgorithm).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static String translateLibraryWrapperFile(InnerDataStructures.Library library, CICAlgorithmType mAlgorithm){
		int index = 0;
		String content = new String();
		
		content += "\n#include \"" + library.getName() + ".h\"\n";
		content += "#include \"" + library.getName() + "_data_structure.h\"\n";
		content += "\n#include \"LIB_port.h\"\n\n";
		content += "#define LIBCALL(x, ...) LIBCALL_##x(__VA_ARGS__)\n";
		
		String t_libCon = new String();
		for(TaskLibraryConnectionType libCon: mAlgorithm.getLibraryConnections().getTaskLibraryConnection()){
			if(libCon.getSlaveLibrary().equals(library.getName())){
				t_libCon = libCon.getMasterPort();
				content += "#define LIBCALL_" + libCon.getMasterPort() + "(f, ...) l_" + library.getName() 
				           + "_##f(__VA_ARGS__)\n\n";
				break;
			}
		}
		
		for(LibraryLibraryConnectionType libCon: mAlgorithm.getLibraryConnections().getLibraryLibraryConnection()){
			if(libCon.getSlaveLibrary().equals(library.getName())){
				t_libCon = libCon.getMasterPort();
				content += "#define LIBCALL_" + libCon.getMasterPort() + "(f, ...) l_" + library.getName() 
				           + "_##f(__VA_ARGS__)\n\n";
				break;
			}
		}
		
		content += "static int write_channel_id, read_channel_id;\n\n";
		content += "void " + library.getName() + "_wrapper_init()\n{\n";
		content += "\twrite_channel_id = init_lib_port(" + library.getIndex() + ", 'w');\n";
		content += "\tread_channel_id = init_lib_port(" + library.getIndex() + ", 'r');\n";
		content += "\tLIBCALL(" + t_libCon + ", init);\n}\n\n";
		
		content += "void " + library.getName() + "_wrapper_go()\n{\n";
		content += "\tint func_index;\n";
		content += "\t" + library.getName() + "_func_data receive_data;\n";
		content += "\t" + library.getName() + "_ret_data send_data;\n";
		
		for(Function func: library.getFuncList()){
			if(func.getReturnType().equalsIgnoreCase("void"))	content += "\tint ret_" + func.getFunctionName() + " = 0;\n";
			else	content += "\t" + func.getReturnType() + " ret_" + func.getFunctionName() + ";\n";
		}
		
		content += "\n\twhile(1)\n\t{\n";
		content += "\t\tLIB_RECEIVE(read_channel_id, &receive_data, sizeof(" + library.getName() + "_func_data));\n";
		content += "\n\t\tfunc_index = receive_data.func_num;";
		content += "\n\t\tsend_data.func_num = func_index;\n\n";
		content += "\t\tswitch(func_index) {\n";
		
		for(Function func: library.getFuncList()){
			content += "\t\t\tcase " + func.getIndex() + " : \n";
			if(func.getReturnType().equalsIgnoreCase("void")){
				content += "\t\t\t\tLIBCALL(" + t_libCon + ", " + func.getFunctionName();
				
				for(Argument arg: func.getArgList())
					content += ", receive_data.func." + func.getFunctionName() + "." + arg.getVariableName();
				content += ");\n";
			}
			else{
				content += "\t\t\t\tret_" + func.getFunctionName() + "= LIBCALL(" + t_libCon + ", " + func.getFunctionName();
				
				for(Argument arg: func.getArgList())
					content += ", receive_data.func." + func.getFunctionName() + "." + arg.getVariableName();
				content += ");\n";
				content += "\t\t\t\tsend_data.ret.ret_" + func.getFunctionName() + " = ret_" + func.getFunctionName() + ";\n";
				content += "\t\t\t\tLIB_SEND(write_channel_id, &send_data, sizeof(" + library.getName() + "_ret_data));\n";
			}
			content += "\t\t\t\tbreak; \n\n";
		}
		content += "\t\t}\n";
		content += "\t}\n}\n\n";
		
		content += "void " + library.getName() + "_wrapper_wrapup()\n{\n";
		content += "\tLIBCALL(" + t_libCon + ", wrapup);\n}\n\n";
		
		return content;
	}
	

	public static void generateLibraryDataStructureHeader(File fileOut, InnerDataStructures.Library library)
	{
		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(translateLibraryDataStructureHeader(library).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
				e.printStackTrace();
		}
	}
	
	public static String translateLibraryDataStructureHeader(InnerDataStructures.Library library){
		int index = 0;
		String content = new String();
		
		content += "#ifndef _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n";
		content += "#define _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n\n";
		content += "#include <stdint.h>\n\n";
		
		for(String extraHeader: library.getExtraHeader())
			content += "#include \"" + extraHeader + "\"\n\n";
		
		content += "// Added by jhw at 10.01.21 for library\n" + "typedef struct {\n" + 
		         "\tint task_num;\n\tint func_num;\n" + "\tunion {\n";

		for(Function func: library.getFuncList()){
			if(func.getArgList().size() > 0){
				content += "\t\tstruct {\n";
				for(Argument arg: func.getArgList()){
					content += "\t\t\t" + arg.getType() + " " + arg.getVariableName() + ";\n";
					index++;
				}
				if(index == 0)	content += "\t\t\tint temp;\n";
				content += "\t\t} " + func.getFunctionName() + ";\n";
			}
		}
		content += "\t} func;\n";
		content += "} " + library.getName() + "_func_data;\n\n";
		
		index = 0;
		content += "typedef struct {\n" + "\tint task_num;\n\tint func_num;\n" + "\tunion {\n";
		for(Function func: library.getFuncList()){
			if(func.getReturnType().equals("void"))	content += "\t\tint ret_" + func.getFunctionName() + ";\n";
			else		content += "\t\t" + func.getReturnType() + " ret_" + func.getFunctionName() + ";\n";
			index++;
		}
		if(index == 0)	content += "\t\t\tint temp;\n";
		
		content += "\t} ret;\n";
		content += "} " + library.getName() + "_ret_data;\n\n";
		content += "\n#endif\n";
	    	    
		return content;
	}
	
	public static void generateLibraryDataStructureHeader_16bit(File fileOut, InnerDataStructures.Library library)
	{
		try {
			FileOutputStream outstream = new FileOutputStream(fileOut);
			outstream.write(translateLibraryDataStructureHeader_16bit(library).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
				e.printStackTrace();
		}
	}
	
	public static String translateLibraryDataStructureHeader_16bit(InnerDataStructures.Library library){
		int index = 0;
		String content = new String();
		
		content += "#ifndef _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n";
		content += "#define _" + library.getName().toUpperCase() + "_DATA_STRUCTURE_H_\n\n";
		content += "#include <stdint.h>\n\n";
		
		for(String extraHeader: library.getExtraHeader())
			content += "#include \"" + extraHeader + "\"\n\n";
		
		content += "// Added by jhw at 10.01.21 for library\n" + "typedef struct {\n" + 
		         "\tlong task_num;\n\tlong func_num;\n" + "\tunion {\n";

		for(Function func: library.getFuncList()){
			if(func.getArgList().size() > 0){
				content += "\t\tstruct {\n";
				for(Argument arg: func.getArgList()){
					String type = "";
					if(arg.getType().equals("int"))		type = "long";
					content += "\t\t\t" + type + " " + arg.getVariableName() + ";\n";
					index++;
				}
				if(index == 0)	content += "\t\t\tlong temp;\n";
				content += "\t\t} " + func.getFunctionName() + ";\n";
			}
		}
		content += "\t} func;\n";
		content += "} " + library.getName() + "_func_data;\n\n";
		
		index = 0;
		content += "typedef struct {\n" + "\tlong task_num;\n\tlong func_num;\n" + "\tunion {\n";
		for(Function func: library.getFuncList()){
			if(func.getReturnType().equals("void"))	content += "\t\tlong ret_" + func.getFunctionName() + ";\n";
			else{
				String retType = "";
				if(func.getReturnType().equals("int"))	retType = "long";
				content += "\t\t" + retType + " ret_" + func.getFunctionName() + ";\n";
			}
			index++;
		}
		if(index == 0)	content += "\t\t\tlong temp;\n";
		
		content += "\t} ret;\n";
		content += "} " + library.getName() + "_ret_data;\n\n";
		content += "\n#endif\n";
	    	    
		return content;
	}


	public static void generateLibraryChannelHeader(String file, String mTemplateFile, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		File fileOut = new File(file);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateLibraryChannelHeader(content, mStubList, mWrapperList).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static String translateLibraryChannelHeader(String mContent, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		String code = mContent;
		String headerIncludeCode="";
		String channelEntriesCode="";
		
		for(InnerDataStructures.Library library: mStubList)
			headerIncludeCode += "#include \"" + library.getName()+ "_data_structure.h\"\n";
		for(InnerDataStructures.Library library: mWrapperList)
			headerIncludeCode += "#include \"" + library.getName()+ "_data_structure.h\"\n";

		code = code.replace("##HEADER_INCLUDE", headerIncludeCode);
		
		return code;
	}
	
	public static void generateLibraryChannelDef(String file, String mTemplateFile, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		File fileOut = new File(file);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateLibraryChannelDef(content, mStubList, mWrapperList).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static String translateLibraryChannelDef(String mContent, ArrayList<InnerDataStructures.Library> mStubList, ArrayList<InnerDataStructures.Library> mWrapperList)
	{
		String code = mContent;
		String channelEntriesCode="";
		
		for(InnerDataStructures.Library library: mStubList){
			channelEntriesCode += "\t{" + library.getIndex() + ", \"" + library.getName() + "\", 'w', NULL, NULL, NULL, sizeof(" + library.getName() + "_func_data), 0, sizeof(" + library.getName() + "_func_data), 0, MUTEX_INIT_INLINE, COND_INIT_INLINE},\n";
			channelEntriesCode += "\t{" + library.getIndex() + ", \"" + library.getName() + "\", 'r', NULL, NULL, NULL, sizeof(" + library.getName() + "_ret_data), 0, sizeof(" + library.getName() + "_ret_data), 0, MUTEX_INIT_INLINE, COND_INIT_INLINE},\n";
		}
		for(InnerDataStructures.Library library: mWrapperList){
			channelEntriesCode += "\t{" + library.getIndex() + ", \"" + library.getName() + "\", 'r', NULL, NULL, NULL, sizeof(" + library.getName() + "_func_data), 0, sizeof(" + library.getName() + "_func_data), 0, MUTEX_INIT_INLINE, COND_INIT_INLINE},\n";
			channelEntriesCode += "\t{" + library.getIndex() + ", \"" + library.getName() + "\", 'w', NULL, NULL, NULL, sizeof(" + library.getName() + "_ret_data), 0, sizeof(" + library.getName() + "_ret_data), 0, MUTEX_INIT_INLINE, COND_INIT_INLINE},\n";
		}

		code = code.replace("##LIBCHANNEL_ENTRIES", channelEntriesCode);
		
		return code;
	}
	
	public static void generateLibraryWrapperDef(String file, String mTemplateFile, ArrayList<InnerDataStructures.Library> mWrapperList, ArrayList<InnerDataStructures.Library> mStubList)
	{
		File fileOut = new File(file);
		File templateFile = new File(mTemplateFile);
		try {
			FileInputStream instream = new FileInputStream(templateFile);
			FileOutputStream outstream = new FileOutputStream(fileOut);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			String content = new String(buffer);
			
			outstream.write(translateLibraryWrapperDef(content, mWrapperList, mStubList).getBytes());	
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static String translateLibraryWrapperDef(String mContent, ArrayList<InnerDataStructures.Library> mWrapperList, ArrayList<InnerDataStructures.Library> mStubList)
	{
		String code = mContent;
		String externFuncDeclCode="";
		String wrapperEntriesCode="";
		int index = 0;
		
		for(InnerDataStructures.Library library: mWrapperList){
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_init(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_go(void);\n";
			externFuncDeclCode += "extern void " + library.getName() + "_wrapper_wrapup(void);\n";
			
			wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", \"" + library.getName() + "_wrapper\", " + library.getName() + "_wrapper_init, "
					+ library.getName() + "_wrapper_go, " + library.getName() + "_wrapper_wrapup, 0},\n";
		}
		
		if(mStubList != null){
			for(InnerDataStructures.Library library: mStubList)
				wrapperEntriesCode += "\t{" + index++ + ", " + library.getProc() +  ", NULL, NULL, NULL, 0},\n";
		}
		
		code = code.replace("##EXTERN_FUNC_DECL", externFuncDeclCode);
		code = code.replace("##LIBWRAPPER_ENTRIES", wrapperEntriesCode);
		
		return code;
	}
}
