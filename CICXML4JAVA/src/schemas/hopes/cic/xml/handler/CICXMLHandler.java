package hae.peace.container.cic.mapping.xml;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.StringWriter;
import hopes.cic.exception.CICXMLException;

public abstract class CICXMLHandler {
	protected abstract void storeResource(StringWriter writer) throws CICXMLException;
	protected abstract void loadResource(ByteArrayInputStream is) throws CICXMLException;
	
	public void setXMLString(String xmlString) throws CICXMLException{
		ByteArrayInputStream is = new ByteArrayInputStream(xmlString.getBytes());
		loadResource(is);
	}
	
	public String getXMLString() throws CICXMLException{
		StringWriter writer = new StringWriter();
		storeResource(writer);
		writer.flush();
		return writer.toString();	
	}
	
	public void storeXMLString(String fileName) throws CICXMLException {
		StringWriter writer = new StringWriter();
		storeResource(writer);
		writer.flush();
		
		FileOutputStream os;
		try {
			os = new FileOutputStream(fileName);
			byte[] abContents = writer.toString().getBytes();
			os.write(abContents, 0, abContents.length);
			os.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public String getLocalFile(String filename) {
		File file = new File(filename);
		if (!file.exists())
			return null;

		try {
			FileInputStream fis = new FileInputStream(file);
			byte[] buffer = new byte[fis.available()];
			fis.read(buffer);
			fis.close();
			return new String(buffer);
		} catch (IOException e) {
			return null;
		}
	}
	
	public boolean putLocalFile(String filename, String data) {
		File file = new File(filename);
		try {
			file.createNewFile();
			FileOutputStream fos = new FileOutputStream(file);
			fos.write(data.getBytes());
			fos.close();
			return true;
		} catch (IOException e) {
			return false;
		}
	}
	
	public void LoadXMLfileToHandler(String xmlFileName) throws CICXMLException {
		String xmlData = getLocalFile(xmlFileName);

		if (xmlData == null || xmlData.indexOf(xmlFileName + " doesn't exist") == 0) {
			throw new CICXMLException(null, "[ERROR] file doesn't exist");
		}
		setXMLString(xmlData);
	}	
}