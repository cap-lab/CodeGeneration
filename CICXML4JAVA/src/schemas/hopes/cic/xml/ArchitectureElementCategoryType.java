
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;


/**
 * <p>Java class for ArchitectureElementCategoryType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ArchitectureElementCategoryType">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="processor"/>
 *     &lt;enumeration value="memory"/>
 *     &lt;enumeration value="dma"/>
 *     &lt;enumeration value="hwip"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlEnum
public enum ArchitectureElementCategoryType {

    @XmlEnumValue("dma")
    DMA("dma"),
    @XmlEnumValue("hwip")
    HWIP("hwip"),
    @XmlEnumValue("memory")
    MEMORY("memory"),
    @XmlEnumValue("processor")
    PROCESSOR("processor");
    private final String value;

    ArchitectureElementCategoryType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ArchitectureElementCategoryType fromValue(String v) {
        for (ArchitectureElementCategoryType c: ArchitectureElementCategoryType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v.toString());
    }

}
