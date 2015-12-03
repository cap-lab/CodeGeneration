package CommonLibraries;

import java.io.*;

public class Util {
	public static Boolean fileIsLive(String isLivefile) {
		File f1 = new File(isLivefile);
	   
		if(f1.exists()){
			return true;
		}
		else{
			return false;
	    }
	}
	
	public static void copyFile(String theDestFile, String theSrcFile)
	{
		try {
			// open template
			File srcFile = new File(theSrcFile);		
			FileInputStream instream = new FileInputStream(srcFile);
			byte[] buffer = new byte[instream.available()];
			instream.read(buffer);
			instream.close();
			
			File destFile = new File(theDestFile);
			FileOutputStream outstream = new FileOutputStream(destFile);		
			outstream.write(buffer);			
			outstream.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void copyExtensionFiles(String mOutputPath, String mSrcPath, String extension)
	{
		File dir = new File(mSrcPath);
		File[] files = dir.listFiles();
		if(dir == null)			return;
				
		for(File file: files){
			if(file.getName().endsWith(extension)){	
				// open template
				try {
					File srcFile = new File(mSrcPath + file.getName());
					FileInputStream instream = new FileInputStream(srcFile);	
					byte[] buffer = new byte[instream.available()];
					instream.read(buffer);
					instream.close();
					
					File destFile = new File(mOutputPath + file.getName());
					FileOutputStream outstream = new FileOutputStream(destFile);
					outstream.write(buffer);			
					outstream.close();
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}	
	}
	
	public static void copyFiles(File targetLocation, File sourceLocation) throws IOException {
		String[] children = sourceLocation.list();
		for (int i=0; i<children.length; i++){
			File f = new File(sourceLocation + "/" + children[i]);
			File t = new File(targetLocation + "/" + children[i]);	
			if(!f.isDirectory()){	
				InputStream in = new FileInputStream(f);
				OutputStream out = new FileOutputStream(t);
	
				byte[] buf = new byte[1024];
				int len;
				while ((len = in.read(buf)) > 0) {
					out.write(buf, 0, len);
				}
				in.close();
				out.close();
				f.delete();
			}
		}
	}
	
	public static void moveFiles(File targetLocation, File sourceLocation) throws IOException {
		String[] children = sourceLocation.list();
		for (int i=0; i<children.length; i++){
			File f = new File(sourceLocation + "/" + children[i]);
			File t = new File(targetLocation + "/" + children[i]);	
			if(!f.isDirectory()){	
				InputStream in = new FileInputStream(f);
				OutputStream out = new FileOutputStream(t);
	
				byte[] buf = new byte[1024];
				int len;
				while ((len = in.read(buf)) > 0) {
					out.write(buf, 0, len);
				}
				in.close();
				out.close();
				f.delete();
			}
		}
	}

	public static void copyAllFiles(File targetLocation, File sourceLocation) throws IOException {
		if (sourceLocation.isDirectory()) {
			if (!targetLocation.exists()) 
				targetLocation.mkdir();

			String[] children = sourceLocation.list();
			for (int i=0; i<children.length; i++)
				copyAllFiles(new File(targetLocation, children[i]), new File(sourceLocation, children[i]));
		} 

		else {
			InputStream in = new FileInputStream(sourceLocation);
			OutputStream out = new FileOutputStream(targetLocation);

			byte[] buf = new byte[1024];
			int len;
			while ((len = in.read(buf)) > 0) {
				out.write(buf, 0, len);
			}
			in.close();
			out.close();
		}
	}
	
	public static String getCodeFromTemplate(String templateFileName, String macro){
		Boolean flag = false;
		String code = "";
		File templateFile = new File(templateFileName);
		try {
			BufferedReader in = new BufferedReader(new FileReader(templateFileName));
		    String line;

		    while ((line = in.readLine()) != null) {
		    	if(line.contains(macro + "_END")){
		    		flag = false;
		    	}
		    	if(flag == true){
		    		code += line + "\n";
		    	}
		    	if(line.contains(macro + "_START")){
		    		flag = true;
		    	}
		    }
		      in.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return code;
	}
	
	public static void deleteFiles(String directory, String extension) {
		 class ExtensionFilter implements FilenameFilter {
			  private String extension;

			  public ExtensionFilter( String extension ) {
			       this.extension = extension;             
			  }
			  public boolean accept(File dir, String name) {
			       return (name.endsWith(extension));
			  }
		 }
	         
	     ExtensionFilter filter = new ExtensionFilter(extension);
	     File dir = new File(directory);

	     String[] list = dir.list(filter);
	     File file;
	     if (list.length == 0) {
	             return;
	     }

	     for (int i = 0; i < list.length; i++) {
	       file = new File(directory + list[i]);
	       boolean isdeleted =   file.delete();
	     }
	}
}
