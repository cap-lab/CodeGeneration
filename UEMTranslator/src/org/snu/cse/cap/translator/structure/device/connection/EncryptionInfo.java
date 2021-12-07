package org.snu.cse.cap.translator.structure.device.connection;

public class EncryptionInfo {
	protected String encryptionType;
	protected String userKey;
	protected String initializationVector;
	protected int userKeyLen;

	public EncryptionInfo(String encryptionType, String userKey) {
		this.encryptionType = encryptionType;
		this.userKey = userKey;
	}

		
	public String getEncryptionType() {
		return encryptionType;
	}
	
	public String getUserKey() {
		return userKey;
	}
	
	public int getUserKeyLen() {
		return userKeyLen;
	}

	public String getInitializationVector() {
		return initializationVector;
	}

	public void setEncryptionType(String encryptiontype) {
		this.encryptionType = encryptiontype;
	}

	public void setUserKey(String userkey) {
		this.userKey = userkey;
	}

	public void setUserKeyLen(int len) {
		this.userKeyLen = len;
	}

	public void setInitializationVector(String initializationvector) {
		this.initializationVector = initializationvector;
	}

}
