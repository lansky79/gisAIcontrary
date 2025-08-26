import React from 'react';
import { 
  Layout, 
  Card, 
  Form, 
  Input, 
  Select, 
  Switch, 
  Slider, 
  Button, 
  Space, 
  Row, 
  Col,
  Divider,
  message,
  Alert
} from 'antd';
import { SaveOutlined, ReloadOutlined, SettingOutlined } from '@ant-design/icons';
import { useAppStore } from '@store/useAppStore';
import type { AnalysisConfig } from '@types/index';

const { Content } = Layout;
const { TextArea } = Input;
const { Option } = Select;

const SettingsPage: React.FC = () => {
  const { analysisConfig, setAnalysisConfig } = useAppStore();
  const [form] = Form.useForm();

  // 初始化表单值
  React.useEffect(() => {
    form.setFieldsValue(analysisConfig);
  }, [analysisConfig, form]);

  // 保存设置
  const handleSave = async () => {
    try {
      const values = await form.validateFields();
      setAnalysisConfig(values);
      message.success('设置已保存');
    } catch (error) {
      message.error('请检查输入内容');
    }
  };

  // 重置为默认值
  const handleReset = () => {
    const defaultConfig: AnalysisConfig = {
      minChangeArea: 100,
      confidenceThreshold: 0.5,
      registrationMethod: 'auto',
      detectionSensitivity: 'medium',
      enableAdvancedFiltering: true,
    };
    
    form.setFieldsValue(defaultConfig);
    setAnalysisConfig(defaultConfig);
    message.success('已重置为默认设置');
  };

  // 获取敏感度说明
  const getSensitivityDescription = (sensitivity: string) => {
    switch (sensitivity) {
      case 'low':
        return '仅检测明显的变化，误报率低但可能遗漏小变化';
      case 'medium':
        return '平衡检测精度和误报率，适合大多数场景';
      case 'high':
        return '检测细微变化，可能产生较多误报但不会遗漏';
      default:
        return '';
    }
  };

  return (
    <Layout style={{ height: '100%', background: '#f5f5f5' }}>
      <Content style={{ padding: 16, overflow: 'auto' }}>
        <Row gutter={16}>
          <Col span={16}>
            <Card 
              title={
                <Space>
                  <SettingOutlined />
                  算法参数设置
                </Space>
              }
              extra={
                <Space>
                  <Button icon={<ReloadOutlined />} onClick={handleReset}>
                    重置默认
                  </Button>
                  <Button type="primary" icon={<SaveOutlined />} onClick={handleSave}>
                    保存设置
                  </Button>
                </Space>
              }
            >
              <Form
                form={form}
                layout="vertical"
                initialValues={analysisConfig}
              >
                <Row gutter={24}>
                  <Col span={12}>
                    <Card title="变化检测参数" size="small" style={{ marginBottom: 16 }}>
                      <Form.Item
                        name="minChangeArea"
                        label="最小变化区域面积"
                        help="小于此面积的变化区域将被过滤（单位：像素）"
                      >
                        <Slider
                          min={50}
                          max={1000}
                          step={10}
                          marks={{
                            50: '50',
                            100: '100',
                            300: '300',
                            500: '500',
                            1000: '1000'
                          }}
                          tooltip={{ formatter: (value) => `${value} 像素` }}
                        />
                      </Form.Item>

                      <Form.Item
                        name="confidenceThreshold"
                        label="置信度阈值"
                        help="低于此置信度的变化区域将被标记为低可信度"
                      >
                        <Slider
                          min={0.1}
                          max={0.9}
                          step={0.05}
                          marks={{
                            0.1: '0.1',
                            0.3: '0.3',
                            0.5: '0.5',
                            0.7: '0.7',
                            0.9: '0.9'
                          }}
                          tooltip={{ formatter: (value) => `${(value * 100).toFixed(0)}%` }}
                        />
                      </Form.Item>

                      <Form.Item
                        name="detectionSensitivity"
                        label="检测敏感度"
                        help={getSensitivityDescription(form.getFieldValue('detectionSensitivity'))}
                      >
                        <Select>
                          <Option value="low">低敏感度</Option>
                          <Option value="medium">中等敏感度</Option>
                          <Option value="high">高敏感度</Option>
                        </Select>
                      </Form.Item>

                      <Form.Item
                        name="enableAdvancedFiltering"
                        label="启用高级过滤"
                        valuePropName="checked"
                      >
                        <Switch
                          checkedChildren="开启"
                          unCheckedChildren="关闭"
                        />
                      </Form.Item>
                      <div style={{ fontSize: 12, color: '#666', marginTop: -16 }}>
                        启用形态学处理和噪声过滤，提升检测质量
                      </div>
                    </Card>
                  </Col>

                  <Col span={12}>
                    <Card title="图像配准参数" size="small" style={{ marginBottom: 16 }}>
                      <Form.Item
                        name="registrationMethod"
                        label="配准算法"
                        help="选择图像配准使用的特征检测算法"
                      >
                        <Select>
                          <Option value="auto">自动选择（推荐）</Option>
                          <Option value="sift">SIFT（精度高，速度慢）</Option>
                          <Option value="surf">SURF（平衡性能）</Option>
                          <Option value="orb">ORB（速度快，精度中等）</Option>
                        </Select>
                      </Form.Item>

                      <Alert
                        message="配准算法说明"
                        description={
                          <ul style={{ margin: 0, paddingLeft: 20 }}>
                            <li><strong>SIFT:</strong> 尺度不变特征变换，精度最高但速度较慢</li>
                            <li><strong>SURF:</strong> 加速稳健特征，速度和精度平衡</li>
                            <li><strong>ORB:</strong> 快速特征检测，速度最快但精度较低</li>
                            <li><strong>自动:</strong> 根据图像特点自动选择最佳算法</li>
                          </ul>
                        }
                        type="info"
                        showIcon
                        style={{ fontSize: 12 }}
                      />
                    </Card>
                  </Col>
                </Row>

                <Divider />

                <Card title="系统配置" size="small">
                  <Row gutter={24}>
                    <Col span={12}>
                      <Form.Item
                        label="API超时时间"
                        help="图像处理API请求的超时时间（秒）"
                      >
                        <Input
                          addonAfter="秒"
                          defaultValue="60"
                          disabled
                        />
                      </Form.Item>

                      <Form.Item
                        label="最大文件大小"
                        help="单个图像文件的最大允许大小"
                      >
                        <Input
                          addonAfter="MB"
                          defaultValue="10"
                          disabled
                        />
                      </Form.Item>
                    </Col>

                    <Col span={12}>
                      <Form.Item
                        label="支持的图像格式"
                        help="系统支持处理的图像文件格式"
                      >
                        <TextArea
                          value="JPEG, PNG, TIFF, JPG, TIF"
                          rows={2}
                          disabled
                        />
                      </Form.Item>

                      <Form.Item
                        label="地图默认中心"
                        help="地图组件的默认中心坐标"
                      >
                        <Input
                          defaultValue="39.9042, 116.4074 (北京)"
                          disabled
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                </Card>
              </Form>
            </Card>
          </Col>

          <Col span={8}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              {/* 当前设置预览 */}
              <Card title="当前设置预览" size="small">
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <div>
                    <strong>最小变化区域:</strong> {analysisConfig.minChangeArea} 像素
                  </div>
                  <div>
                    <strong>置信度阈值:</strong> {(analysisConfig.confidenceThreshold * 100).toFixed(0)}%
                  </div>
                  <div>
                    <strong>检测敏感度:</strong> {
                      analysisConfig.detectionSensitivity === 'low' ? '低' :
                      analysisConfig.detectionSensitivity === 'medium' ? '中' : '高'
                    }
                  </div>
                  <div>
                    <strong>配准算法:</strong> {
                      analysisConfig.registrationMethod === 'auto' ? '自动选择' :
                      analysisConfig.registrationMethod.toUpperCase()
                    }
                  </div>
                  <div>
                    <strong>高级过滤:</strong> {analysisConfig.enableAdvancedFiltering ? '已启用' : '已禁用'}
                  </div>
                </Space>
              </Card>

              {/* 性能影响说明 */}
              <Card title="性能影响说明" size="small">
                <Alert
                  message="设置建议"
                  description={
                    <div style={{ fontSize: 12 }}>
                      <p><strong>高精度场景:</strong> 使用SIFT算法，高敏感度，启用高级过滤</p>
                      <p><strong>实时处理:</strong> 使用ORB算法，中等敏感度，适当降低精度要求</p>
                      <p><strong>平衡模式:</strong> 自动算法选择，中等敏感度，启用高级过滤</p>
                    </div>
                  }
                  type="info"
                  showIcon
                />
              </Card>

              {/* 版本信息 */}
              <Card title="版本信息" size="small">
                <Space direction="vertical" size="small" style={{ width: '100%' }}>
                  <div>
                    <strong>系统版本:</strong> v1.0.0
                  </div>
                  <div>
                    <strong>算法版本:</strong> v2.1.0
                  </div>
                  <div>
                    <strong>最后更新:</strong> 2024-01-15
                  </div>
                </Space>
              </Card>

              {/* 帮助信息 */}
              <Card title="帮助信息" size="small">
                <div style={{ fontSize: 12, color: '#666' }}>
                  <p>如需了解更多算法参数说明，请参考系统文档或联系技术支持。</p>
                  <p>设置修改后立即生效，影响后续的检测任务。</p>
                </div>
              </Card>
            </Space>
          </Col>
        </Row>
      </Content>
    </Layout>
  );
};

export default SettingsPage;