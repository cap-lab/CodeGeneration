/*
 *
 * Copyright (c) 2009 by CAPLAB
 * All rights reserved.
 *
 * This software is the confidential and proprietary information
 * of CAPLAB("Confidential Information"). You
 * shall not disclose such Confidential Information and shall use
 * it only in accordance with the terms of the license agreement
 * you entered into with CAPLAB.
 */

package about.hopes.cic.xml;

import java.util.HashMap;
import java.util.Map;


/**
 *
 * 
 * @author long21s
 */

public class BuildNumber {
    private static final String LIBRARY_NAME = "CIC XML JAXB Binded API";
    private static final String DESCRIPTION = "CAPLAB CIC XML JAXB Binded API";
    private static final String VERSION_NUMBER = "1.0.7";
    private static final String OTHER_INFO = "CICControl added\n"
    	+ "RunConditionType->run-once added\n";
    private static String BUILD_NUMBER = "0007";
    private static String BUILD_DATE = "2009/04/13 09:00";
    

    private static final String LIBRARY_INFO = DESCRIPTION + "\nVersion: " + VERSION_NUMBER
            + " [" + BUILD_NUMBER + " (" + BUILD_DATE + ")]\n" + OTHER_INFO;

    private static Map<String, String> keywordMap;
    
    static {
        keywordMap = new HashMap<String, String>();
        keywordMap.put("name", LIBRARY_NAME);
        keywordMap.put("description", DESCRIPTION);
        keywordMap.put("version", VERSION_NUMBER);
        keywordMap.put("build", BUILD_NUMBER);
        keywordMap.put("date", BUILD_DATE);
        keywordMap.put("info", OTHER_INFO);
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println(LIBRARY_INFO);
        } else {
            for (String arg : args) {
                String value = keywordMap.get(arg.toLowerCase());
                if (value == null) {
                    System.err.print("Unknown keyword '" + arg + "'");
                } else {
                    System.out.print(value + " ");
                }
            }
            System.out.println("");
        }
    }
}
